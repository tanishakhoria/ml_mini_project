"""Live demo: train (or load) the paper's network and classify sample test records.

Usage:  python src/demo.py            # 12 random test records
        python src/demo.py --n 20 --seed 7
First run trains for ~20 s and caches the model in results/demo_model.joblib.
"""
import argparse
import os

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import run_experiment as rx

CACHE = os.path.join(rx.OUT, "demo_model.joblib")


def build():
    tr, te = rx.load()
    Xtr, Xte, _ = rx.preprocess(tr, te, StandardScaler)
    known = sorted(set(tr["label"]) - {"normal"})
    classes = ["normal"] + known + ["other"]
    ytr = tr["label"].values
    yte = np.where(te["label"].isin(["normal"] + known), te["label"], "other")
    if os.path.exists(CACHE):
        model = joblib.load(CACHE)
    else:
        print("Training 122-500-24 network (10 epochs)...")
        model = MLPClassifier(hidden_layer_sizes=(500,), activation="logistic", solver="sgd",
                              learning_rate_init=0.5, momentum=0.0, nesterovs_momentum=False,
                              batch_size=1000, alpha=1e-4, random_state=0)
        for _ in range(10):
            model.partial_fit(Xtr, ytr, classes=classes)
        joblib.dump(model, CACHE)
    return model, Xte, yte, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    model, Xte, yte, te = build()
    pred = model.predict(Xte)
    print(f"\nFull test set: 24-class accuracy {np.mean(pred == yte):.1%}, "
          f"Normal/Abnormal accuracy {np.mean((pred == 'normal') == (yte == 'normal')):.1%}\n")
    idx = np.random.default_rng(a.seed).choice(len(Xte), a.n, replace=False)
    print(f"{'#':>3} {'protocol':8} {'service':10} {'true':16} {'predicted':16} verdict")
    for k, i in enumerate(idx, 1):
        row = te.iloc[i]
        verdict = "ALERT" if pred[i] != "normal" else "ok"
        mark = "" if (pred[i] == "normal") == (yte[i] == "normal") else "  <-- wrong"
        print(f"{k:>3} {row['protocol_type']:8} {row['service']:10} {yte[i]:16} {pred[i]:16} {verdict}{mark}")


if __name__ == "__main__":
    main()
