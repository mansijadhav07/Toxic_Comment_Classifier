# Toxic Comment Classification

An extended research notebook for multi-label toxic-comment classification. The notebook starts with a TF-IDF and logistic-regression baseline, then provides optional experiments for validation, imbalance handling, model comparison, robustness, calibration, explainability, and a Gradio demo.

## Repository contents

- `toxic_comment_classifier_final.ipynb` — the project notebook.

Datasets are not included. The notebook is designed for Google Colab, but its data paths can be changed to match another environment.

## Getting started

1. Open `toxic_comment_classifier_final.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter.
2. Obtain the data files described below and make them accessible to the notebook.
3. In **Section 2: Config**, update the paths to those files.
4. Run the notebook from the beginning. Optional sections are controlled by `RUN_*` flags in the configuration; enable only the experiments you want and have the required data and packages for.

The default paths in the notebook point to `/content/drive/MyDrive/toxic data/`. If using Google Drive, mount Drive in Colab before running the data-loading section, or replace the paths with locations available in your runtime.

## Data

The core workflow expects:

| File | Purpose | Expected columns |
| --- | --- | --- |
| `train.csv` | Training comments and labels | `id`, `comment_text`, and the six label columns below |
| `test.csv` | Comments to classify | `id`, `comment_text` |
| `sample_submission.csv` | Submission template | `id` and the six label columns |

The six labels are `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. By default, the notebook expects these labels in `train.csv`. If the training file does not contain them, provide `test_labels.csv` via `TRAIN_LABEL_PATH`; the notebook uses its labels for training.

Additional datasets support optional experiments:

- `davidson_tweets.csv` — cross-domain evaluation.
- `jigsaw_bias_identity.csv` — subgroup and bias evaluation.
- `cad_contextual_raw.csv` — contextual/adversarial data experiments.

These are not required for the basic training and prediction workflow. Update or disable the related configuration flags if you do not have them.

## Experiments and output

The notebook includes sections for:

- TF-IDF/logistic-regression baseline and multi-label metrics.
- Stratified cross-validation and class-imbalance strategies.
- Optional BiLSTM and transformer fine-tuning (DistilBERT by default).
- Calibration, uncertainty, efficiency, quantization, and distillation.
- Explainability, adversarial robustness, bias/subgroup analysis, and optional LLM benchmarking.
- A Gradio interface for trying the trained model.

The final prediction section writes `submission.csv` to the notebook's current working directory. Some optional analyses also write CSV reports there.

## Dependencies

The notebook uses Python packages including pandas, NumPy, scikit-learn, matplotlib, and seaborn. Optional sections may require additional packages such as PyTorch, Transformers, iterative-stratification, SHAP, or Gradio. Some notebook cells install packages when their corresponding sections are run; if running locally, install any missing package in your environment.

## API keys

The Gemini benchmark is optional. If you enable it, provide `GOOGLE_API_KEY` through Colab Secrets or an environment variable. Do not paste credentials into notebook cells, save them in notebook outputs, or commit them to the repository. If a key has been exposed, revoke or rotate it.
