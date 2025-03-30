## Machine Learning Project Structure

```
Machine-Learning-CI-CD/
│── data/                  # Raw and processed data
│   ├── raw/               # Unprocessed data
│   ├── processed/         # Processed data for training/testing
│
│── notebooks/             # Jupyter notebooks for experiments
│
│── src/                   # Source code for training and evaluation
│   ├── models/            # Model architecture scripts
│   ├── preprocessing/     # Data cleaning and transformation scripts
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation and metrics scripts
│
│── config/                # Configuration files
│   ├── hp_config.json     # Hyperparameter tuning configuration
│   ├── dvc.yaml           # Data versioning configuration
│
│── reports/               # Generated reports and plots
│   ├── metrics.json       # Evaluation metrics
│   ├── confusion_matrix.png # Performance visualization
│
│── scripts/               # Utility and automation scripts
│   ├── hp_tuning.py       # Hyperparameter tuning script
│   ├── metrics_and_plots.py # Script for metrics and plots
│
│── .gitignore             # Git ignore file
│── .dvcignore             # DVC ignore file
│── README.md              # Documentation about the project
