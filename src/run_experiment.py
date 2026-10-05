"""Neural-network based security attack detection on NSL-KDD.

Part 1: reproduction of the reference paper's network (122-500-24, sigmoid/softmax).
Part 2: extensions - a tuned ReLU MLP and a Random Forest baseline.

Trains an MLP (the neural network) and a Random Forest (baseline) and reports
5-class (Normal/DoS/Probe/R2L/U2R) and binary (Normal vs Attack) results.

Usage:  python src/run_experiment.py            (from the repo root)
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files",
        "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]

ATTACK_GROUP = {"normal": "Normal"}
for a in ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
          "udpstorm", "processtable", "worm", "mailbomb"]:
    ATTACK_GROUP[a] = "DoS"
for a in ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]:
    ATTACK_GROUP[a] = "Probe"
for a in ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
          "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
          "sendmail", "named"]:
    ATTACK_GROUP[a] = "R2L"
for a in ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack",
          "xterm", "ps", "httptunnel"]:
    ATTACK_GROUP[a] = "U2R"
CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def load():
    tr = pd.read_csv(os.path.join(DATA, "KDDTrain+.txt"), names=COLS)
    te = pd.read_csv(os.path.join(DATA, "KDDTest+.txt"), names=COLS)
    for d in (tr, te):
        d["group"] = d["label"].map(ATTACK_GROUP)
        d.drop(columns=["difficulty"], inplace=True)
    assert tr["group"].notna().all() and te["group"].notna().all()
    return tr, te


def preprocess(tr, te, scaler_cls=MinMaxScaler):
    cat = ["protocol_type", "service", "flag"]
    both = pd.concat([tr.assign(_s="tr"), te.assign(_s="te")])
    both = pd.get_dummies(both, columns=cat)  # shared columns -> no train/test mismatch
    tr_x = both[both["_s"] == "tr"].drop(columns=["_s", "label", "group"])
    te_x = both[both["_s"] == "te"].drop(columns=["_s", "label", "group"])
    scaler = scaler_cls().fit(tr_x)  # fit on train only
    return (scaler.transform(tr_x).astype("float32"),
            scaler.transform(te_x).astype("float32"), list(tr_x.columns))


def evaluate(name, model, Xtr, ytr, Xte, yte, labels, results, tag):
    # integer-encode labels (sklearn's MLP early stopping fails on string labels)
    idx = {l: i for i, l in enumerate(labels)}
    ytr_i = np.array([idx[v] for v in ytr])
    t0 = time.time()
    model.fit(Xtr, ytr_i)
    fit_s = time.time() - t0
    pred = np.array(labels)[model.predict(Xte)]
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    results[f"{name} | {tag}"] = {"accuracy": acc, "macro_f1": f1, "train_seconds": fit_s}
    print(f"[{tag}] {name}: acc={acc:.4f} macroF1={f1:.4f} ({fit_s:.1f}s)")
    print(classification_report(yte, pred, labels=labels, zero_division=0))
    cm = confusion_matrix(yte, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{name} ({tag})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"cm_{name.replace(' ', '_')}_{tag}.png"), dpi=150)
    plt.close(fig)
    return model


def paper_reproduction(tr, te, results, n_seeds=10):
    """Reproduce the reference paper's network (CS229 2021):
    122 inputs -> 500 sigmoid hidden -> 24 softmax outputs
    (normal + 22 known attacks + 'other' for attacks unseen in training),
    mini-batch SGD, lr 0.5, batch 1000, 10 epochs, standardised inputs.
    Reports 24-class accuracy and Normal/Abnormal accuracy over several seeds
    (the paper notes results vary a lot with random initialisation)."""
    Xtr, Xte, _ = preprocess(tr, te, StandardScaler)
    known = sorted(set(tr["label"]) - {"normal"})
    assert len(known) == 22, len(known)
    classes = ["normal"] + known + ["other"]  # 24 outputs
    ytr = tr["label"].values
    yte = np.where(te["label"].isin(["normal"] + known), te["label"], "other")
    n_ab_te = yte != "normal"

    def make(seed):
        return MLPClassifier(hidden_layer_sizes=(500,), activation="logistic",
                             solver="sgd", learning_rate_init=0.5, momentum=0.0,
                             nesterovs_momentum=False, batch_size=1000,
                             alpha=1e-4, random_state=seed)

    def scores(m):
        ptr, pte = m.predict(Xtr), m.predict(Xte)
        return (accuracy_score(ytr, ptr), accuracy_score(yte, pte),
                np.mean((pte != "normal") == n_ab_te))

    runs = []
    for seed in range(n_seeds):
        m = make(seed)
        for _ in range(10):
            m.partial_fit(Xtr, ytr, classes=classes)
        a_tr, a_te, a_bin = scores(m)
        runs.append({"seed": seed, "train_acc": a_tr, "test_acc_24class": a_te,
                     "test_acc_normal_vs_abnormal": a_bin})
        print(f"[paper repro] seed {seed}: train={a_tr:.4f} test24={a_te:.4f} normal/abnormal={a_bin:.4f}")
    te24 = np.array([r["test_acc_24class"] for r in runs])
    teb = np.array([r["test_acc_normal_vs_abnormal"] for r in runs])
    results["paper_reproduction"] = {
        "runs": runs,
        "test_acc_24class": {"mean": te24.mean(), "std": te24.std(), "min": te24.min(), "max": te24.max()},
        "test_acc_normal_vs_abnormal": {"mean": teb.mean(), "std": teb.std(), "min": teb.min(), "max": teb.max()}}
    print(f"[paper repro] 24-class test acc {te24.mean():.4f} +/- {te24.std():.4f}; "
          f"normal/abnormal {teb.mean():.4f} +/- {teb.std():.4f}")

    # accuracy-vs-epoch curves (paper Figure 2), 50 epochs, seed 0
    m = make(0)
    tr_curve, te_curve = [], []
    for _ in range(50):
        m.partial_fit(Xtr, ytr, classes=classes)
        a_tr, a_te, _ = scores(m)
        tr_curve.append(a_tr); te_curve.append(a_te)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    ax[0].plot(tr_curve, color="red"); ax[0].set_title("(a) Train")
    ax[1].plot(te_curve, color="blue"); ax[1].set_title("(b) Test")
    for a in ax:
        a.set_xlabel("epoch"); a.set_ylabel("accuracy")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "paper_repro_accuracy_vs_epoch.png"), dpi=150); plt.close(fig)
    results["paper_reproduction"]["curve_seed0"] = {"train": tr_curve, "test": te_curve}


def main():
    np.random.seed(SEED)
    tr, te = load()
    print("train", tr.shape, "test", te.shape)
    print(tr["group"].value_counts().to_string(), "\n")
    pd.DataFrame({"train": tr["group"].value_counts(),
                  "test": te["group"].value_counts()}).to_csv(os.path.join(OUT, "class_distribution.csv"))
    Xtr, Xte, feats = preprocess(tr, te)
    print("features after encoding:", len(feats))

    results = {}
    paper_reproduction(tr, te, results)
    mk_mlp = lambda: MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                                   solver="adam", alpha=1e-4, batch_size=256,
                                   learning_rate_init=1e-3, max_iter=60,
                                   early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=8, random_state=SEED)
    mk_rf = lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)

    # 5-class
    mlp = evaluate("Neural Network (MLP)", mk_mlp(), Xtr, tr["group"], Xte, te["group"], CLASSES, results, "5-class")
    evaluate("Random Forest", mk_rf(), Xtr, tr["group"], Xte, te["group"], CLASSES, results, "5-class")

    # binary
    ybin_tr = np.where(tr["group"] == "Normal", "Normal", "Attack")
    ybin_te = np.where(te["group"] == "Normal", "Normal", "Attack")
    evaluate("Neural Network (MLP)", mk_mlp(), Xtr, ybin_tr, Xte, ybin_te, ["Normal", "Attack"], results, "binary")
    evaluate("Random Forest", mk_rf(), Xtr, ybin_tr, Xte, ybin_te, ["Normal", "Attack"], results, "binary")

    # NN training curve (5-class model)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(mlp.loss_curve_, label="train loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(); ax.set_title("MLP training loss")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "mlp_loss_curve.png"), dpi=150); plt.close(fig)

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
