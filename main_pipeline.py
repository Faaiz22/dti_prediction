
import argparse
from pathlib import Path
import torch

# Import all modules from the backend
from backend.utils.config import DTIConfig
from backend.utils.logger import setup_logger
from backend.utils.visualization import (
    plot_training_history, plot_performance_metrics, create_summary_report
)
from backend.data_acquisition.enricher import DataEnricher
from backend.featurization.molecular import MolecularFeaturizer
from backend.featurization.protein import ProteinEmbedder
from backend.featurization.dataset import create_dataloaders
from backend.featurization.splitting import scaffold_split, plot_scaffold_distribution, random_split
from backend.modeling.dti_model import DTIModel
from backend.training.dti_trainer import DTITrainer
from backend.inference.dti_inference import DTIPredictor
from backend.inference.dti_explainability import DTIExplainer

def train_pipeline(config: DTIConfig):
    """Complete training pipeline."""
    logger = setup_logger("main_pipeline", level=config.log_level)
    logger.info("="*70)
    logger.info("STARTING DTI TRAINING PIPELINE")
    logger.info("="*70)

    # Step 1: Data Acquisition
    logger.info("\n[1/7] Data Acquisition and Enrichment")
    enricher = DataEnricher(config)
    df = enricher.create_dataset()

    # Step 2: Data Splitting
    logger.info("\n[2/7] Dataset Splitting")
    if config.use_scaffold_split:
        train_df, val_df, test_df = scaffold_split(df, config)
        plot_scaffold_distribution(train_df, val_df, test_df, save_path=config.output_dir / "scaffold_distribution.png")
    else:
        train_df, val_df, test_df = random_split(df, config)

    # Step 3: Featurization
    logger.info("\n[3/7] Molecular and Protein Featurization")
    mol_featurizer = MolecularFeaturizer(config)
    prot_embedder = ProteinEmbedder(config)

    # Step 4: Create DataLoaders
    logger.info("\n[4/7] Creating DataLoaders")
    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, config, mol_featurizer, prot_embedder)

    # Step 5: Model Initialization
    logger.info("\n[5/7] Initializing Model")
    model = DTIModel(config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized with {trainable_params:,} trainable parameters.")

    # Step 6: Training
    logger.info("\n[6/7] Training Model")
    trainer = DTITrainer(model, train_loader, val_loader, config, test_loader)
    history = trainer.train()
    best_model_path = trainer.save_best_model()
    plot_training_history(history, config.output_dir / "training_history.png")

    # Step 7: Evaluation
    logger.info("\n[7/7] Final Evaluation on Test Set")
    test_metrics = trainer.evaluate(test_loader, "Test")
    plot_performance_metrics(test_metrics["labels"], test_metrics["predictions"], save_path=config.output_dir / "test_performance.png")

    train_metrics = trainer.evaluate(train_loader, "Train")
    val_metrics = trainer.evaluate(val_loader, "Validation")
    report = create_summary_report(train_metrics, val_metrics, test_metrics, config.to_dict(), save_path=config.output_dir / "evaluation_report.txt")
    print(report)

    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info("="*70)

def predict_pipeline(smiles: str, sequence: str, model_path: Path, config: DTIConfig):
    """Prediction pipeline for a single drug-target pair."""
    logger = setup_logger("prediction_pipeline", level=config.log_level)
    logger.info("="*70)
    logger.info("STARTING DTI PREDICTION PIPELINE")

    logger.info("\n[1/4] Loading Model and Featurizers")
    model, _ = DTIModel.load_checkpoint(model_path, config.device)
    mol_featurizer = MolecularFeaturizer(config)
    prot_embedder = ProteinEmbedder(config)

    logger.info("\n[2/4] Making Prediction")
    predictor = DTIPredictor(model, config, mol_featurizer, prot_embedder)
    result = predictor.predict(smiles, sequence, return_uncertainty=True, return_attention=True)

    logger.info("\n[3/4] Generating Explanation")
    explainer = DTIExplainer(model, config)
    graph = mol_featurizer.smiles_to_graph(smiles)
    protein_emb = prot_embedder.embed(sequence)
    explanation = explainer.explain_prediction(graph, protein_emb, smiles, sequence)

    logger.info("\n[4/4] Generating Report")
    report = explainer.generate_report(explanation, save_path=config.output_dir / "prediction_explanation.txt")

    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print(f"SMILES: {smiles}")
    print(f"Target: {sequence[:50]}...")
    print(f"Binding Score: {result['score']:.4f}")
    if 'uncertainty' in result:
        ci = result['confidence_interval']
        print(f"Uncertainty (Std Dev): {result['uncertainty']:.4f}")
        print(f"95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"Prediction: {'BINDS' if result['score'] >= 0.5 else 'DOES NOT BIND'}")
    print("="*70)
    print("\n" + report)

def main():
    parser = argparse.ArgumentParser(description="DTI Prediction Pipeline")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], required=True)
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--smiles", type=str, help="Drug SMILES string (for predict mode)")
    parser.add_argument("--sequence", type=str, help="Protein sequence (for predict mode)")
    parser.add_argument("--model", type=str, help="Path to model checkpoint (for predict mode)")
    args = parser.parse_args()

    config = DTIConfig.load(Path(args.config)) if args.config else DTIConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        train_pipeline(config)
    elif args.mode == "predict":
        model_path = Path(args.model) if args.model else config.checkpoint_dir / "best_model_final.pt"
        predict_pipeline(args.smiles, args.sequence, model_path, config)

if __name__ == "__main__":
    main()
