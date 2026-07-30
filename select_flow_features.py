import argparse
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import RobustScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_csv", default="dataset/ar002_et12_20260511_002-stage1_flows.csv")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--topk", type=int, default=30)
    args = parser.parse_args()

    df = pd.read_csv(args.flow_csv)
    df.columns = [str(c).strip() for c in df.columns]

    drop_cols = [
        "flow_id",
        "flow_start_timestamp_us",
        "flow_end_timestamp_us",
        "source_ip",
        "destination_ip",
        "source_port",
        "destination_port",
        args.label_col,
    ]

    feature_cols = [
        c for c in df.columns
        if c not in drop_cols
    ]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[args.label_col].astype(int).values

    # 只保留数值列
    X = X.select_dtypes(include=[np.number])
    feature_cols = list(X.columns)

    # log1p for non-negative heavy-tailed features
    X = X.clip(lower=0)
    X = np.log1p(X.values)

    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=130,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=130,
    )

    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("val f1_label1:", f1_score(y_val, preds, pos_label=1))
    print("val pr_auc:", average_precision_score(y_val, probs))

    importance = clf.feature_importances_
    order = np.argsort(importance)[::-1]

    top_features = [feature_cols[i] for i in order[:args.topk]]

    print("\nTop features:")
    for f in top_features:
        print(f"  - {f}")


if __name__ == "__main__":
    main()