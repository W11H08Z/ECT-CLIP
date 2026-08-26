import os
import glob
import json
import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import tqdm
import monai.transforms as transforms
import monai.data as data

# Offline-stage artifacts (sentence embeddings, itemized reports) shipped with
# the repository. Everything that does not depend on raw CT volumes is read
# from this directory by default.
OFFLINE_STAGE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "offline_stage_data",
)


def load_sent2embeddings(sent2embeddings_path_list: List[str]) -> Dict[str, np.ndarray]:
    """Load and merge one or more sentence->embedding checkpoints."""
    res = {}
    for sent2embeddings_path in sent2embeddings_path_list:
        sent2embeddings = torch.load(sent2embeddings_path, map_location="cpu")
        for sent, embedding in sent2embeddings.items():
            if sent not in res:
                res[sent] = embedding.numpy()
            else:
                res[sent] = np.concatenate([res[sent], embedding.numpy()])
    return res


class CTRATEDataset(data.PersistentDataset):
    """
    CT-RATE training dataset with text-level negative sampling.

    Two negative-sampling strategies are supported:

    - ``negative_sampling="random"`` (default): negatives are uniformly sampled
      from all sentences that do not belong to the current report. Requires no
      entity annotations.
    - ``negative_sampling="hard"``: hard negatives are sentences that share at
      least one UMLS CUI (anatomy or clinical phenomenon) with the current
      report but do NOT share an exact (anatomy, clinical_phenomenon) CUI pair
      (exact matches are potential false negatives). This strategy requires the
      three entity-linking JSON files:
        * sent_entities_json: sentence -> extracted entities
        * anatomy_entity_links_json: anatomy entity -> CUI
        * clinical_phenomenon_entity_links_json: clinical phenomenon entity -> CUI
    """

    def __init__(
        self,
        data_folder: str,
        labels_csv: str,
        itemized_text_json: str = os.path.join(
            OFFLINE_STAGE_DATA_DIR, "Itemized_medical_reports.json"
        ),
        sent2embeddings_path_list: Optional[List[str]] = None,
        text_feat_cnt: int = 256,
        negative_sampling: str = "random",
        hard_negative_target: int = 25,
        sent_entities_json: Optional[str] = None,
        anatomy_entity_links_json: Optional[str] = None,
        clinical_phenomenon_entity_links_json: Optional[str] = None,
    ):
        if negative_sampling not in ("random", "hard"):
            raise ValueError(
                f"negative_sampling must be 'random' or 'hard', got {negative_sampling!r}"
            )
        if negative_sampling == "hard" and not all(
            p is not None
            for p in (
                sent_entities_json,
                anatomy_entity_links_json,
                clinical_phenomenon_entity_links_json,
            )
        ):
            raise ValueError(
                "negative_sampling='hard' requires sent_entities_json, "
                "anatomy_entity_links_json and clinical_phenomenon_entity_links_json. "
                "Either provide these files or use negative_sampling='random'."
            )

        if sent2embeddings_path_list is None:
            sent2embeddings_path_list = [
                os.path.join(OFFLINE_STAGE_DATA_DIR, "CXR_sent2embedding.pt")
            ]

        self.data_folder = data_folder
        self.labels_csv = labels_csv
        self.text_feat_cnt = text_feat_cnt
        self.negative_sampling = negative_sampling
        self.hard_negative_target = hard_negative_target

        # Load metadata (all from offline_stage_data by default)
        self.sent2embeddings = load_sent2embeddings(sent2embeddings_path_list)
        self.text_dim = list(self.sent2embeddings.values())[0].shape[0]
        self.accession_to_text_list = self._load_json(itemized_text_json)

        # Entity-linking data (optional; only needed for hard negative mining)
        self.sent_entities = self._load_json(sent_entities_json)
        self.anatomy_entity2cui = self._load_json(anatomy_entity_links_json)
        self.clinical_phenomenon_entity2cui = self._load_json(
            clinical_phenomenon_entity_links_json
        )
        self.entity_data_available = all(
            d is not None
            for d in (
                self.sent_entities,
                self.anatomy_entity2cui,
                self.clinical_phenomenon_entity2cui,
            )
        )

        self.samples = self._prepare_samples()
        random.shuffle(self.samples)

        self.transform = transforms.Compose([
            # Load NIfTI for the "image" key only, ensure channel-first (C, H, W, D)
            transforms.LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0, a_max=1.0,      # input is [0, 1]
                b_min=-1.0, b_max=1.0,     # output to [-1, 1]
                clip=True,
            ),
            transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=(112, 256, 352)),
            transforms.RandSpatialCropd(keys=["image"], roi_size=(96, 240, 336), random_size=False),
            transforms.ToTensord(keys=["image"]),
        ])

        super().__init__(data=self.samples, transform=self.transform, cache_dir=None)

        # Per-sentence CUI cache (cheap: a few dict lookups per sentence).
        self.sent_cui_cache = {
            sent: self._get_cuis_for_sentence(sent) for sent in self.sent2embeddings
        }
        self.all_sentences_set = set(self.sent2embeddings.keys())

        # CUI -> sentences inverted indexes, only needed for hard negative mining.
        self.ana_cui_to_sents: Dict[str, set] = {}
        self.clin_cui_to_sents: Dict[str, set] = {}
        self.cui_pair_to_sents: Dict[tuple, set] = {}
        if self.negative_sampling == "hard":
            self._build_cui_indexes()

    # ------------------------------------------------------------------ #
    # Loading helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Load a JSON file, or return None when no path is provided."""
        if path is None:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_cuis_for_sentence(self, sentence: str) -> tuple:
        """
        Retrieve (anatomy_cui, clinical_phenomenon_cui) for a given sentence.
        Falls back to 'CUI_NONE' when entity-linking data is unavailable or the
        mapping is missing.
        """
        if not self.entity_data_available:
            return "CUI_NONE", "CUI_NONE"
        entities = self.sent_entities.get(sentence, {})
        ana_cui = self.anatomy_entity2cui.get(entities.get("organ_locations", ""), "CUI_NONE")
        clin_cui = self.clinical_phenomenon_entity2cui.get(
            entities.get("abnormal_findings", ""), "CUI_NONE"
        )
        ana_cui = ana_cui if ana_cui is not None else "CUI_NONE"
        clin_cui = clin_cui if clin_cui is not None else "CUI_NONE"
        return ana_cui, clin_cui

    def _build_cui_indexes(self):
        """Build inverted indexes from CUI (or CUI pair) to sentences."""
        for sent, (ana_cui, clin_cui) in self.sent_cui_cache.items():
            if ana_cui != "CUI_NONE":
                self.ana_cui_to_sents.setdefault(ana_cui, set()).add(sent)
            if clin_cui != "CUI_NONE":
                self.clin_cui_to_sents.setdefault(clin_cui, set()).add(sent)
            if ana_cui != "CUI_NONE" and clin_cui != "CUI_NONE":
                self.cui_pair_to_sents.setdefault((ana_cui, clin_cui), set()).add(sent)

    # ------------------------------------------------------------------ #
    # Sample preparation
    # ------------------------------------------------------------------ #
    def _prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, "*"))

        df = pd.read_csv(self.labels_csv)
        label_cols = [c for c in df.columns if c != "VolumeName"]
        df[label_cols] = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        label_dict = df.set_index("VolumeName")[label_cols].to_dict("index")

        for patient_folder in tqdm.tqdm(patient_folders):
            accession_folders = glob.glob(os.path.join(patient_folder, "*"))
            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, "*.nii.gz"))

                for nii_file in nii_files:
                    file_name = os.path.basename(nii_file)
                    acc_id = "_".join(file_name.split("_")[:3])

                    if acc_id not in self.accession_to_text_list:
                        continue
                    if file_name not in label_dict:
                        continue

                    official_labels = list(label_dict[file_name].values())
                    text_list, text_feats, pos_label = [], [], []
                    anatomy_cuis, clinical_phenomenon_cuis = [], []
                    for sent in self.accession_to_text_list[acc_id]:
                        text_list.append(sent)
                        text_feats.append(self.sent2embeddings[sent])
                        pos_label.append(1)

                        ana_cui, clin_cui = self._get_cuis_for_sentence(sent)
                        anatomy_cuis.append(ana_cui)
                        clinical_phenomenon_cuis.append(clin_cui)

                    if len(text_list) == 0:
                        continue

                    samples.append({
                        "image": nii_file,
                        "text_list": text_list,
                        "official_labels": official_labels,
                        "pos_label": pos_label,
                        "text_feats": text_feats,
                        "accession": file_name,
                        "anatomy_cui": anatomy_cuis,
                        "clinical_phenomenon_cui": clinical_phenomenon_cuis,
                    })
        return samples

    # ------------------------------------------------------------------ #
    # Negative sampling
    # ------------------------------------------------------------------ #
    def _sample_random_negatives(self, positive_sents: set, num_needed: int) -> List[str]:
        """
        Random negative sampling (default strategy).

        Negatives are drawn uniformly from every sentence in the corpus that
        does not appear in the current report. If the pool is smaller than
        ``num_needed``, the remainder is drawn with replacement.
        """
        pool = list(self.all_sentences_set - positive_sents)
        if len(pool) >= num_needed:
            return random.sample(pool, num_needed)

        selected = pool[:]
        while len(selected) < num_needed:
            selected.append(random.choice(pool))
        return selected

    def _split_negative_pools(self, text_list, anatomy_cuis, clinical_phenomenon_cuis):
        """
        Split the corpus into three mutually exclusive pools w.r.t. one report:

        - exact_match_set: sentences sharing an exact (anatomy, clinical) CUI
          pair with the report. These are potential false negatives, so they are
          excluded from negatives.
        - hard_negative_set: sentences sharing at least one CUI with the report
          but no exact CUI pair.
        - random_negative_set: everything else outside the report.
        """
        report_sentences_set = set(text_list)
        report_anatomy_cuis = set(c for c in anatomy_cuis if c != "CUI_NONE")
        report_clinical_cuis = set(c for c in clinical_phenomenon_cuis if c != "CUI_NONE")
        report_cui_pairs = set(zip(anatomy_cuis, clinical_phenomenon_cuis))

        exact_match_set = set()
        for pair in report_cui_pairs:
            exact_match_set.update(self.cui_pair_to_sents.get(pair, set()))
        exact_match_set -= report_sentences_set

        hard_candidate_set = set()
        for ana_cui in report_anatomy_cuis:
            hard_candidate_set.update(self.ana_cui_to_sents.get(ana_cui, set()))
        for clin_cui in report_clinical_cuis:
            hard_candidate_set.update(self.clin_cui_to_sents.get(clin_cui, set()))
        hard_negative_set = hard_candidate_set - report_sentences_set - exact_match_set

        random_negative_set = (
            self.all_sentences_set
            - report_sentences_set
            - exact_match_set
            - hard_negative_set
        )
        return report_sentences_set, exact_match_set, hard_negative_set, random_negative_set

    def _sample_hard_negatives(
        self, text_list, anatomy_cuis, clinical_phenomenon_cuis, num_needed
    ) -> List[str]:
        """
        Hard negative sampling.

        Take up to ``hard_negative_target`` hard negatives first; fill the rest
        from the random pool; fall back to sampling with replacement from all
        non-positive sentences if both pools are exhausted.
        """
        (
            report_sentences_set,
            exact_match_set,
            hard_negative_set,
            random_negative_set,
        ) = self._split_negative_pools(text_list, anatomy_cuis, clinical_phenomenon_cuis)

        hard_pool = list(hard_negative_set)
        hard_target = min(self.hard_negative_target, num_needed)
        selected = (
            random.sample(hard_pool, hard_target)
            if len(hard_pool) >= hard_target
            else hard_pool[:]
        )

        remaining = num_needed - len(selected)
        if remaining > 0:
            random_pool = list(random_negative_set)
            if len(random_pool) >= remaining:
                selected += random.sample(random_pool, remaining)
            else:
                selected += random_pool[:]
                # Fallback: draw with replacement from all non-positive sentences
                all_non_positive = list(
                    self.all_sentences_set - report_sentences_set - exact_match_set
                )
                while len(selected) < num_needed:
                    selected.append(random.choice(all_non_positive))

        return selected

    # ------------------------------------------------------------------ #
    # Dataset interface
    # ------------------------------------------------------------------ #
    def __getitem__(self, index):
        data_dict = super().__getitem__(index)

        text_list = data_dict["text_list"][:]
        text_feats = [self.sent2embeddings[s] for s in text_list]
        pos_label = [1.0] * len(text_list)
        anatomy_cuis = data_dict["anatomy_cui"][:]
        clinical_phenomenon_cuis = data_dict["clinical_phenomenon_cui"][:]

        num_needed = self.text_feat_cnt - len(text_list)
        if num_needed > 0:
            if self.negative_sampling == "hard":
                selected_negs = self._sample_hard_negatives(
                    text_list, anatomy_cuis, clinical_phenomenon_cuis, num_needed
                )
            else:
                selected_negs = self._sample_random_negatives(set(text_list), num_needed)

            for sent in selected_negs:
                text_list.append(sent)
                text_feats.append(self.sent2embeddings[sent])
                pos_label.append(0.0)
                ana_cui, clin_cui = self.sent_cui_cache[sent]
                anatomy_cuis.append(ana_cui)
                clinical_phenomenon_cuis.append(clin_cui)

        # Safety trim if somehow we exceeded text_feat_cnt
        if len(text_list) > self.text_feat_cnt:
            trim = len(text_list) - self.text_feat_cnt
            text_list = text_list[:-trim]
            text_feats = text_feats[:-trim]
            pos_label = pos_label[:-trim]
            anatomy_cuis = anatomy_cuis[:-trim]
            clinical_phenomenon_cuis = clinical_phenomenon_cuis[:-trim]

        return {
            "image": data_dict["image"],
            "text_list": text_list,
            "official_labels": np.array(data_dict["official_labels"], dtype=np.float32),
            "pos_label": np.array(pos_label, dtype=np.float32),
            "text_feats": np.array(text_feats, dtype=np.float32),
            "accession": data_dict["accession"],
            "anatomy_cui": anatomy_cuis,
            "clinical_phenomenon_cui": clinical_phenomenon_cuis,
        }

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    """
    Smoke test. Image volumes and the labels CSV are NOT part of
    offline_stage_data; point data_folder / labels_csv to your local CT-RATE
    copies before running:

        python datasets/training_dataset.py <data_folder> <labels_csv>
    """
    import sys
    from torch.utils.data import DataLoader

    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    data_folder, labels_csv = sys.argv[1], sys.argv[2]

    print("Initializing dataset (random negative sampling)...")
    dataset = CTRATEDataset(
        data_folder=data_folder,
        labels_csv=labels_csv,
        negative_sampling="hard",
        text_feat_cnt=256,
    )
    print(f"Dataset length: {len(dataset)}, text dim: {dataset.text_dim}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    for batch_idx, batch in enumerate(dataloader):
        print(
            f"Batch {batch_idx}: "
            f"image {tuple(batch['image'].shape)}, "
            f"text_feats {tuple(batch['text_feats'].shape)}, "
            f"positives {int(batch['pos_label'].sum().item())}"
        )
        if batch_idx >= 4:
            break
