import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import monai.transforms as transforms

PATHOLOGIES = [
        # 'Medical material',
        'Arterial wall calcification',
        'Cardiomegaly',
        'Pericardial effusion',
        'Coronary artery wall calcification',
        'Hiatal hernia',
        # 'Lymphadenopathy',
        'Emphysema',
        'Atelectasis',
        'Lung nodule',
        'Lung opacity',
        'Pulmonary fibrotic sequela',
        'Pleural effusion',
        'Mosaic attenuation pattern',
        'Peribronchial thickening',
        'Consolidation',
        'Bronchiectasis',
        'Interlobular septal thickening'
    ]

PROMPTS = [
        ["Not {disease}", "{disease}"],
        ["Ruled out {disease}", "Confirmed {disease}"],
        ["No findings of {disease}", "Findings suggestive of {disease}"],
        ["Negative for {disease}", "Positive for {disease}"],
        ["No evidence of {disease}", "Findings consistent with {disease}"],
        ["No sign of {disease}", "Sign of {disease}"],
        ["No indication of {disease}", "Indication of {disease}"],
        ["{disease} is not present", "{disease} is present"],
        ["Absence of {disease}", "Presence of {disease}"],
        ["No diagnosis of {disease}", "Diagnosis of {disease}"],
        ["No imaging findings of {disease}", "Imaging findings of {disease}"],
        ["No features of {disease}", "Features of {disease}"],
        ["No manifestation of {disease}", "Manifestation of {disease}"],
    ]

# Default text-encoder checkpoints (Hugging Face hub ids, or local paths).
DEFAULT_ENCODER_PATHS = {
    "CXR": "microsoft/BiomedVLP-CXR-BERT-specialized",
    "BioLORD": "FremyCompany/BioLORD-2023",
    "RadBERT": "zzxslp/RadBERT-RoBERTa-4m",
}

TEXT_ENCODER_DIMS = {
    "CXR": 768,
    "BioLORD": 768,
    "RadBERT": 768,
    "MedCLIP": 512,
}


class DataFolder(Dataset):
    """
    CT-RATE validation images with labels and reports.

    NOTE: none of these files are part of offline_stage_data/; point the paths
    to your local CT-RATE copies.
    """

    def __init__(self, data_folder, labels_csv, reports_csv):
        super().__init__()

        self.img_paths = self._load_img_paths(data_folder)
        self.label_dict = self._load_label_csv_to_dict(labels_csv)
        self.reports_dict = self._load_reports_csv_to_dict(reports_csv)

        self.transform = transforms.Compose([
            # Load NIfTI for the "image" key only, ensure channel-first (C, H, W, D)
            transforms.LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0, a_max=1.0,      # input is [0, 1]
                b_min=-1.0, b_max=1.0,     # output to [-1, 1]
                clip=True
            ),
            transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=(112, 256, 352)),
            transforms.CenterSpatialCropd(keys=["image"], roi_size=(96, 240, 336)),
            transforms.ToTensord(keys=["image"])
        ])

    def __len__(self):
        return len(self.img_paths)

    def _load_img_paths(self, vis_root):
        img_paths = []
        for root, _, files in os.walk(vis_root):
            for file in files:
                if file.endswith('.nii.gz'):
                    img_paths.append(os.path.join(root, file))
        return img_paths

    def _load_label_csv_to_dict(self, csv_file):
        df = pd.read_csv(csv_file)
        label_cols = list(df.columns[1:])
        label_dict = {row['VolumeName']: row[label_cols].values.astype("float32") for _, row in df.iterrows()}
        return label_dict

    def _load_reports_csv_to_dict(self, csv_file):
        df = pd.read_csv(csv_file)
        return {row['VolumeName']: self._sanitize_text(f"{row['Findings_EN'] or ''}{row['Impressions_EN'] or ''}") for _, row in df.iterrows()}

    def _sanitize_text(self, s):
        return s.replace('"', '').replace("'", '').replace('(', '').replace(')', '')

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        image = self.transform({'image': image_path})['image']

        acc_num = os.path.basename(image_path)
        label = self.label_dict[acc_num]
        report = self.reports_dict[acc_num]

        return image, report, label, acc_num


# Known text_dim -> encoder combination, used to infer the training-time
# encoder setup directly from the checkpoint.
TEXT_DIM_TO_ENCODERS = {
    768: ["CXR"],                            # any single 768-dim encoder
    512: ["MedCLIP"],
    1536: ["CXR", "BioLORD"],                # two 768-dim encoders
    2304: ["CXR", "BioLORD", "RadBERT"],     # three 768-dim encoders
}


def load_model(ckpt_path):
    """
    Load an ECTCLIP model, inferring text_dim from the checkpoint weights.

    Supports both checkpoint formats:
    - full training checkpoint: {"model": state_dict, "optimizer": ..., ...}
    - bare state_dict saved via torch.save(model.state_dict(), ...)
    """
    from models.ectclip import ECTCLIP

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    # proj_x4 maps text_dim -> 512, so its input dim IS the training text_dim.
    # Fall back to i2t_proj (512 -> text_dim) if proj_x4 is missing.
    if "proj_x4.weight" in state_dict:
        text_dim = state_dict["proj_x4.weight"].shape[1]
    elif "i2t_proj.weight" in state_dict:
        text_dim = state_dict["i2t_proj.weight"].shape[0]
    else:
        raise KeyError(
            f"Cannot infer text_dim from {ckpt_path}: "
            "neither 'proj_x4.weight' nor 'i2t_proj.weight' found. "
            f"Available keys: {list(state_dict.keys())[:5]} ..."
        )

    model = ECTCLIP(text_dim)
    model.load_state_dict(state_dict, strict=True)
    return model


def CXR_text_encode(texts, model_path, device="cuda"):
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
    encoder = BertModel.from_pretrained(model_path).to(device)

    inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = encoder(**inputs)
        text_feats = outputs.last_hidden_state[:, 0, :]

    return text_feats.cpu().numpy()


def BioLORD_text_encode(texts, model_path, device="cuda"):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoder = AutoModel.from_pretrained(model_path).to(device)

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = encoder(**inputs)
        text_feats = mean_pooling(outputs, inputs['attention_mask'])

    return text_feats.cpu().numpy()


def MedCLIP_text_encode(texts, device="cuda"):
    from medclip import MedCLIPModel, MedCLIPVisionModelViT
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()

    encoder = model.text_model.to(device)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_feats = encoder(inputs["input_ids"], inputs["attention_mask"])

    return text_feats.cpu().numpy()


def RadBERT_text_encode(texts, model_path, device="cuda"):
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoder = AutoModel.from_pretrained(model_path, config=config).to(device)

    inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = encoder(**inputs)
        text_feats = outputs.last_hidden_state[:, 0, :]

    return text_feats.cpu().numpy()


def encode_text_features(texts, text_encoders, encoder_paths, device="cuda"):
    """Concatenate text features in the given encoder order."""
    text_feats = []
    if "CXR" in text_encoders:
        text_feats.append(CXR_text_encode(texts, encoder_paths["CXR"], device))
    if "BioLORD" in text_encoders:
        text_feats.append(BioLORD_text_encode(texts, encoder_paths["BioLORD"], device))
    if "RadBERT" in text_encoders:
        text_feats.append(RadBERT_text_encode(texts, encoder_paths["RadBERT"], device))
    if "MedCLIP" in text_encoders:
        text_feats.append(MedCLIP_text_encode(texts, device))

    if len(text_feats) == 0:
        raise ValueError(
            f"No text encoder selected: {text_encoders}. "
            f"Available: {list(TEXT_ENCODER_DIMS.keys())}"
        )

    return np.concatenate(text_feats, axis=-1)


def zeroshot(model, data_folder, labels_csv, reports_csv, save_dir,
             text_encoders, encoder_paths):
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = DataFolder(data_folder, labels_csv, reports_csv)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=16, shuffle=False)

    texts = []
    for prompts in PROMPTS:
        for pathology in PATHOLOGIES:
            texts.append(prompts[1].format(disease=pathology))

    text_feats = encode_text_features(texts, text_encoders, encoder_paths, device)
    text_feats = torch.from_numpy(text_feats).float().to(device)
    # ECTCLIP.forward expects text_feats with a batch dim: (1, N, text_dim)
    text_feats = text_feats.unsqueeze(0)

    model = model.to(device)
    results = {f"prompt_version_{i}": [] for i in range(len(PROMPTS))}
    for images, _, onehotlabels, acc_name in tqdm(dataloader):
        images = images.to(device)
        with torch.no_grad():
            image_feats, text_feats_out = model(images, text_feats)
            logits = (image_feats * text_feats_out).sum(dim=-1)
            probs = torch.sigmoid(logits)

        probs = probs.reshape(len(PROMPTS), len(PATHOLOGIES))

        for i in range(len(PROMPTS)):
            prob = probs[i, :].cpu().tolist()
            results[f"prompt_version_{i}"].append([acc_name[0]] + prob)

    for i in range(len(PROMPTS)):
        save_path = f"{save_dir}/ctrate_prompt_version_{i}.csv"
        pd.DataFrame(
            results[f"prompt_version_{i}"],
            columns=['file_name'] + [f'{k}_{k}' for k in PATHOLOGIES]
        ).to_csv(save_path, index=False, encoding='utf-8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot CT-RATE evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save output CSV files")
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Folder with validation .nii.gz volumes (NOT part of offline_stage_data)")
    parser.add_argument("--labels_csv", type=str, required=True,
                        help="CSV with validation labels (VolumeName + label columns)")
    parser.add_argument("--reports_csv", type=str, required=True,
                        help="CSV with validation reports (VolumeName, Findings_EN, Impressions_EN)")
    parser.add_argument(
        "--text_encoders",
        type=str,
        default=None,
        help="Comma-separated text encoders used during training, e.g. CXR,BioLORD,RadBERT. "
             "If not given, inferred from the text_dim stored in the checkpoint."
    )
    parser.add_argument("--cxr_bert_path", type=str, default=DEFAULT_ENCODER_PATHS["CXR"],
                        help="HF hub id or local path of the CXR-BERT checkpoint")
    parser.add_argument("--biolord_path", type=str, default=DEFAULT_ENCODER_PATHS["BioLORD"],
                        help="HF hub id or local path of the BioLORD checkpoint")
    parser.add_argument("--radbert_path", type=str, default=DEFAULT_ENCODER_PATHS["RadBERT"],
                        help="HF hub id or local path of the RadBERT checkpoint")
    args = parser.parse_args()

    # text_dim always comes from the checkpoint itself
    model = load_model(args.checkpoint)
    actual_text_dim = model.proj_x4.in_features

    if args.text_encoders is not None:
        text_encoders = [name.strip() for name in args.text_encoders.split(",") if name.strip()]
        if not text_encoders:
            raise ValueError(
                f"No text encoders given. Available encoders: {', '.join(TEXT_ENCODER_DIMS.keys())}"
            )
        # Check that the given encoders match the checkpoint's text_dim
        expected_text_dim = sum(TEXT_ENCODER_DIMS[name] for name in text_encoders)
        if expected_text_dim != actual_text_dim:
            raise ValueError(
                f"Text encoder dimensions do not match checkpoint: "
                f"encoders {text_encoders} give {expected_text_dim}, "
                f"but checkpoint has text_dim={actual_text_dim}."
            )
    else:
        if actual_text_dim not in TEXT_DIM_TO_ENCODERS:
            raise ValueError(
                f"Cannot infer text encoders from checkpoint text_dim={actual_text_dim}. "
                f"Known dims: {list(TEXT_DIM_TO_ENCODERS.keys())}. "
                "Please pass --text_encoders explicitly."
            )
        text_encoders = TEXT_DIM_TO_ENCODERS[actual_text_dim]
        print(f"Inferred text encoders from checkpoint (text_dim={actual_text_dim}): {text_encoders}")

    encoder_paths = {
        "CXR": args.cxr_bert_path,
        "BioLORD": args.biolord_path,
        "RadBERT": args.radbert_path,
    }

    zeroshot(model, args.data_folder, args.labels_csv, args.reports_csv,
             args.save_dir, text_encoders, encoder_paths)
