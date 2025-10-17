
# Drug-Target Interaction Prediction with EGNN

This project implements a deep learning pipeline to predict the binding affinity between a drug molecule and a protein target, featuring an E(n)-Equivariant GNN and a Streamlit-based user interface.

## Key Features

- **Model**: E(n)-Equivariant Graph Neural Network (EGNN) for 3D-aware molecular representation, combined with a cross-attention mechanism.
- **Explainability**: Utilizes Integrated Gradients (IG) via Captum to provide atom-level importance scores.
- **Uncertainty Quantification**: Implements Monte Carlo (MC) Dropout to estimate model confidence.
- **Web Interface**: A user-friendly app built with Streamlit for easy prediction and visualization.

## Project Structure

```
dti_prediction/
├── backend/
│   ├── data_acquisition/
│   ├── featurization/
│   ├── inference/
│   ├── modeling/
│   ├── training/
│   └── utils/
├── frontend/
│   ├── components/
│   ├── interface/
│   └── utils/
├── checkpoints/
├── data/
├── results/
├── app.py
├── main_pipeline.py
└── requirements.txt
```

## Setup and Usage

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone <your-repo-url>
cd dti_prediction
pip install -r requirements.txt
```

### 2. Running the Streamlit App

Place your trained model checkpoint in the `checkpoints/` directory (e.g., `checkpoints/best_model_final.pt`).
```bash
streamlit run app.py
```

### 3. Training the Model

Prepare your dataset (e.g., `data/relationships.tsv`) and a configuration file. Then run the training pipeline:
```bash
python main_pipeline.py --mode train --config path/to/your/config.json
```
