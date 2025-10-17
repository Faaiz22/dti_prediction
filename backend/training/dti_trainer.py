
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from tqdm.auto import tqdm
from backend.utils.config import DTIConfig
from backend.utils.logger import setup_logger

class DTITrainer:
    def __init__(self, model, train_loader, val_loader, config: DTIConfig, test_loader=None):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = GradScaler(enabled=config.use_amp)
        self.logger = setup_logger("trainer", level=config.log_level)
        self.best_val_auc = 0.0
        self.early_stop_counter = 0
        self.history = {k: [] for k in ["train_loss", "val_loss", "train_auc", "val_auc", "val_aupr", "val_f1"]}

    def _run_epoch(self, data_loader, is_training: bool):
        self.model.train(is_training)
        total_loss, all_preds, all_labels = 0.0, [], []

        pbar = tqdm(data_loader, desc=f"Epoch {self.current_epoch+1} [{ 'Train' if is_training else 'Val' }]")
        for batch in pbar:
            graph, protein, labels = batch["graph"].to(self.device), batch["protein"].to(self.device), batch["labels"].to(self.device)

            with torch.set_grad_enabled(is_training):
                with autocast(enabled=self.config.use_amp):
                    logits, _ = self.model(graph, protein)
                    loss = self.criterion(logits, labels)

                if is_training:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if is_training: pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(data_loader)
        auc = roc_auc_score(all_labels, all_preds)
        aupr = average_precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
        return {"loss": avg_loss, "auc": auc, "aupr": aupr, "f1": f1}

    def train(self):
        self.logger.info("Starting training...")
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            train_metrics = self._run_epoch(self.train_loader, is_training=True)
            val_metrics = self._run_epoch(self.val_loader, is_training=False)

            self.history["train_loss"].append(train_metrics["loss"]); self.history["train_auc"].append(train_metrics["auc"])
            self.history["val_loss"].append(val_metrics["loss"]); self.history["val_auc"].append(val_metrics["auc"])
            self.history["val_aupr"].append(val_metrics["aupr"]); self.history["val_f1"].append(val_metrics["f1"])

            self.logger.info(f"Epoch {epoch+1}: Train Loss: {train_metrics['loss']:.4f}, Val AUC: {val_metrics['auc']:.4f}, Val AUPR: {val_metrics['aupr']:.4f}")

            if val_metrics["auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc"]
                self.best_model_state = self.model.state_dict()
                self.early_stop_counter = 0
                self.logger.info(f"✓ New best model! Val AUC: {val_metrics['auc']:.4f}")
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.config.early_stopping_patience:
                self.logger.info("Early stopping triggered.")
                break

        if self.best_model_state: self.model.load_state_dict(self.best_model_state)
        return self.history

    def evaluate(self, data_loader, split_name: str = "Test"):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"{split_name} Evaluation"):
                graph, protein, labels = batch["graph"].to(self.device), batch["protein"].to(self.device), batch["labels"].to(self.device)
                logits, _ = self.model(graph, protein)
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        binary_preds = (np.array(all_preds) >= 0.5).astype(int)
        return {
            "auc": roc_auc_score(all_labels, all_preds),
            "aupr": average_precision_score(all_labels, all_preds),
            "accuracy": accuracy_score(all_labels, binary_preds),
            "f1": f1_score(all_labels, binary_preds),
            "predictions": all_preds, "labels": all_labels
        }

    def save_best_model(self):
        if self.best_model_state:
            save_path = self.config.checkpoint_dir / "best_model_final.pt"
            torch.save({"model_state_dict": self.best_model_state, "config": self.config.to_dict()}, save_path)
            self.logger.info(f"Best model saved to {save_path}")
            return save_path
        return None
