# Neural Network for Security Attack Detection (NSL-KDD)
https://github.com/tanishakhoria/ml_mini_project
UE24CS352A Machine Learning mini-project.
Team: Tanisha Khoria PES2UG24CS551 and Sparsha Arun PES2UG24CS513.

We reproduce the paper's neural network on NSL-KDD (Part 1), then extend it with a tuned ReLU MLP and a Random Forest baseline on 5-class and binary tasks (Part 2).

Reference: *Using Neural Network for Security Attacks Detection*, CS229 (Stanford, Spring 2021),
https://cs229.stanford.edu/proj2021spr/report2/81970400.pdf

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/download_data.py     # fetches KDDTrain+.txt and KDDTest+.txt into data/
```

## Run
```bash
python src/run_experiment.py
```
Takes roughly 10-15 minutes on a CPU (10 seeds + a 50-epoch curve for Part 1). Outputs go to `results/`:
`metrics.json`, `class_distribution.csv`, `paper_repro_accuracy_vs_epoch.png`, confusion matrices (`cm_*.png`), `mlp_loss_curve.png`.

## Structure
```
src/download_data.py    fetch dataset
src/run_experiment.py   preprocessing, training, evaluation, plots
results/                generated metrics and figures
data/                   dataset (git-ignored)
```

## Method (short)
- Dataset: NSL-KDD, KDDTrain+ (125,973) / KDDTest+ (22,544), 41 features; one-hot encoding gives 122 features.
- **Part 1 (paper reproduction):** standardised inputs; 24 targets = normal + 22 known attacks + "other" (attacks unseen in training); network 122-500-24, sigmoid hidden, softmax output, mini-batch SGD (lr 0.5, batch 1000, L2 1e-4), 10 epochs; repeated over 10 seeds. Reports 24-class and Normal/Abnormal accuracy.
- **Part 2 (extensions):** Min-Max scaling; ReLU MLP 122-128-64 (Adam, early stopping) and Random Forest (200 trees); attacks grouped into DoS/Probe/R2L/U2R.

## Results (KDDTest+)
| Experiment | Result |
|---|---|
| Paper network, 24-class acc (10 seeds) | 68.6% +/- 0.1 (paper ~70%) |
| Paper network, Normal/Abnormal acc | 75.2% +/- 0.5 (paper ~80%) |
| ReLU MLP vs Random Forest, 5-class acc | 78.5% vs 75.4% |
| ReLU MLP vs Random Forest, binary acc | 81.9% vs 78.4% |
