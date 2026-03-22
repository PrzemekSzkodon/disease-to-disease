# disease-to-disease
Can we predict whether someone has a chronic disease, given their demographics and what other diseases they have? This project builds a multi-task learning (MTL) neural network to do exactly that, using NHANES 2021-23 survey data across 15 conditions.

P(d_k = 1 | x_demo, d_-k)   for k = 1, ..., 15

## Running order
Please run the scripts in the following order:
1. `model.ipynb` trains the MTL network and saves weights, scaler, and test predictions to `saved/`, uses nhanes_clean.csv which is produced from the raw NHANES files and does not require data.ipynb to be run.
2. `evaluation.ipynb` loads all saved artifacts and produces the evaluation figures
3. `baselines.ipynb` trains comparison models against the same test set

* `data.ipynb` is included for transparency only, it documents how `nhanes_clean.csv` was produced from the raw NHANES files and does not need to be run.

To reproduce from scratch, download the 2021-23 NHANES XPT files from the [CDC NHANES portal](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023), place them in a folder `raw_nhanes/` and run from `data.ipynb`. All saved artifacts are already provided in `saved/` so you can skip straight to `model.ipynb`.

## Structure
```
├── data.ipynb # cleaning, feature engineering, saving to saved/
├── model.ipynb # MTL model, training loop, saving weights and predictions
├── evaluation.ipynb # metrics, feature importance, PDPs, ROC/PR curves
├── baselines.ipynb # logistic regression, random forest, STL-ANN, MTL-no-disease
├── saved/
│   ├── nhanes_clean.csv # pre-processed dataset (7,269 participants)
│   ├── col_lists.json # column definitions
│   ├── demo_scaler.pkl # fitted StandardScaler
│   ├── mtl_model.pt # trained model weights
│   ├── mtl_test_preds.npy # test predictions (1091 x 15)
│   └── mtl_test_targets.npy # test targets (1091 x 15)
├── environment.yaml
└── requirements.txt
```

## Setup
```bash
conda env create -f environment.yaml
conda activate d200-mtl
```

or with pip:
```bash
pip install -r requirements.txt
```

## Model
Shared trunk MTL network. 23 demographic inputs go through two hidden layers (128 units, BatchNorm, ReLU, Dropout=0.3) into a shared 128-dimensional representation. This gets concatenated with the disease flags and passed to 15 independent sigmoid heads, one per disease. Trained with a masked BCE loss, Adam (lr=1e-3), early stopping on val PRAUC with patience=20.

Split: 5,088 train / 1,090 val / 1,091 test (70/15/15, random_state=42).

## Report and presentation
The accompanying report and presentation slides are included in the repo root:
1. Report: `Szkodon2025Do_diseases_predict_other_diseases_A_multitask_learning_deep_neural_network_approach_.pdf`
2. Presentation: `Szkodon2025PresentationDo_diseases_predict_diseases.pdf`

## AI Assistance Disclosure
Claude (Anthropic) was used as an AI assistant during development. Specifically, it helped with:
- Debugging training loop and loss function implementation
- Generating baseline comparison code (logistic regression, random forest, STL-ANN, MTL-no-disease)
- Producing evaluation figures (heatmaps, ROC/PR curves, PDP plots)

All model architecture, research design, data pipeline, and analytical decisions were made by the author. Generated code was reviewed, tested, and modified throughout.