import os
from tqdm import tqdm
import swanlab

import torch
from torch.utils.data import DataLoader

from datasets.training_dataset import CTRATEDataset
from models.ectclip import ECTCLIP, SigLIPLoss


def run_supervised_training(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = config.get("save_dir", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    swanlab.init(project="ECT-CLIP", experiment_name="test", config=config)

    # Create Dataset
    # Optional keys left as None fall back to the dataset defaults, which point
    # to the files shipped in offline_stage_data/.
    dataset_kwargs = {
        "data_folder": config["data_folder"],
        "labels_csv": config["labels_csv"],
        "text_feat_cnt": config.get("text_feat_cnt", 256),
        "negative_sampling": config.get("negative_sampling", "random"),
    }
    for key in ("itemized_text_json", "sent2embeddings_path_list",
                "sent_entities_json", "anatomy_entity_links_json",
                "clinical_phenomenon_entity_links_json"):
        if config.get(key) is not None:
            dataset_kwargs[key] = config[key]

    dataset = CTRATEDataset(**dataset_kwargs)
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 1),
        num_workers=config.get("num_workers", 16),
        shuffle=True,
    )

    # Create Model
    model = ECTCLIP(dataset.text_dim)
    model = model.to(device)

    # Create Optimizer
    accumulation_steps = config.get("accumulation_steps", 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("wd", 1e-4),
    )
    num_batches = len(dataloader)
    epochs = config.get("epochs", 20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * num_batches // accumulation_steps, eta_min=1e-6
    )

    criterion = SigLIPLoss().to(device)

    # ================= Training Loop =================
    for epoch in range(epochs):
        model.train()
        iter_bar = tqdm(enumerate(dataloader), total=num_batches, desc=f"Epoch {epoch+1}/{epochs}")

        for i, batch in iter_bar:
            images = batch["image"].to(device)
            text_feats = batch["text_feats"].to(device)
            pos_labels = batch["pos_label"].to(device)

            image_feats, text_feats = model(images, text_feats)

            # 1. Compute SigLIP loss
            # 2. Divide by accumulation_steps for gradient averaging
            total_loss = criterion(image_feats, text_feats, pos_labels) / accumulation_steps
            total_loss.backward()

            # Update parameters when reaching accumulation steps or at the last batch
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(iter_bar):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging: restore the normal loss value
                total_loss = total_loss.item() * accumulation_steps
                swanlab.log({"train_loss": total_loss})
                iter_bar.set_postfix({"loss": f"{total_loss:.4f}"})

        path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, path)
        print(f"Epoch {epoch+1} finished. Saved: {path}")

    print("Training finished.")


if __name__ == "__main__":
    config = {
        # Dataset Setting
        # NOTE: data_folder / labels_csv are NOT part of offline_stage_data;
        # point them to your local CT-RATE copies.
        "data_folder": "path/to/ctrate/processed_train_images",
        "labels_csv": "path/to/ctrate/multi_abnormality_labels/train_predicted_labels.csv",
        # The files below default to the copies shipped in offline_stage_data/;
        # override here only if needed.
        "itemized_text_json": "./offline_stage_data/Itemized_medical_reports.json",
        "sent2embeddings_path_list": [
            "./offline_stage_data/CXR_sent2embedding.pt",
            "./offline_stage_data/BioLORD_sent2embedding.pt",
            "./offline_stage_data/RadBERT_sent2embedding.pt"
        ],
        "text_feat_cnt": 512,
        "negative_sampling": "random",  # or "hard" for hard negative sampling
        "batch_size": 1,
        "num_workers": 16,

        # Optimizer Setting
        "lr": 1e-4,
        "wd": 1e-4,
        "epochs": 5,
        "accumulation_steps": 4,
    }
    config["save_dir"] = "./checkpoints"
    run_supervised_training(config)
