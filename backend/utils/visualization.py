
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score

def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss over Epochs"); axes[0].legend()
    axes[1].plot(history["train_auc"], label="Train AUC")
    axes[1].plot(history["val_auc"], label="Val AUC")
    axes[1].set_title("AUC over Epochs"); axes[1].legend()
    plt.tight_layout()
    if save_path: save_path.parent.mkdir(parents=True, exist_ok=True); plt.savefig(save_path)
    plt.show()

def plot_performance_metrics(labels, predictions, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fpr, tpr, _ = roc_curve(labels, predictions)
    axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc_score(labels, predictions):.3f}')
    axes[0].plot([0, 1], [0, 1], 'r--'); axes[0].set_title("ROC Curve"); axes[0].legend()

    precision, recall, _ = precision_recall_curve(labels, predictions)
    axes[1].plot(recall, precision, label=f'AP = {average_precision_score(labels, predictions):.3f}')
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend()

    cm = confusion_matrix(labels, (np.array(predictions) >= 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path: save_path.parent.mkdir(parents=True, exist_ok=True); plt.savefig(save_path)
    plt.show()

def create_summary_report(train_metrics, val_metrics, test_metrics, config, save_path):
    report = f"EVALUATION REPORT\n{'='*20}\nConfig: {config}\n"
    report += f"Train Metrics: AUC={train_metrics['auc']:.4f}, AUPR={train_metrics['aupr']:.4f}\n"
    report += f"Val Metrics:   AUC={val_metrics['auc']:.4f}, AUPR={val_metrics['aupr']:.4f}\n"
    report += f"Test Metrics:  AUC={test_metrics['auc']:.4f}, AUPR={test_metrics['aupr']:.4f}\n"
    if save_path: save_path.parent.mkdir(parents=True, exist_ok=True); save_path.write_text(report)
    return report
