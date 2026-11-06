from pathlib import Path
from collections import Counter
from datetime import datetime
import pickle
import time

import cv2
import numpy as np
from skimage.feature import hog

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ---------- Configuration ----------

BASE_DIR = Path(__file__).resolve().parent

CIFAR_DIR = (
    BASE_DIR.parent / "cifar-10-batches-py"
)

RESULTS_FILE = BASE_DIR / "knn_results.md"

TRAIN_BATCH_FILES = [
    f"data_batch_{i}" for i in range(1, 6)
]
TEST_BATCH_FILE = "test_batch"
META_FILE = "batches.meta"

CIFAR_IMAGE_SHAPE = (3, 32, 32)  # channels, height, width as stored

IMAGE_SIZE = (64, 64)  # OpenCV uses (width, height)
N_FOLDS = 5
RANDOM_STATE = 42

HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}

# K values compared by cross-validation before the final model is
# refitted on all training data. Only odd values, so a tie between
# two classes cannot happen for a binary-like neighbourhood.
CANDIDATE_K = [1, 3, 5, 7, 9, 11, 13, 15, 21, 25, 31]
CANDIDATE_METRICS = ["euclidean", "manhattan"]

# CIFAR-10 has 50,000 train / 10,000 test images. HOG + brute-force
# KNN grid search over all of them is very slow, so we subsample
# per class. Raise these for a more faithful (but slower) evaluation.
TRAIN_SAMPLES_PER_CLASS = 500
TEST_SAMPLES_PER_CLASS = 100

N_JOBS = -1


# ---------- CIFAR-10 batch loading ----------

def unpickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file, encoding="bytes")


def load_label_names(meta_path):
    meta = unpickle(meta_path)
    return [
        name.decode("utf-8")
        for name in meta[b"label_names"]
    ]


def load_cifar_batch(batch_path):
    """Load one CIFAR-10 batch file into (images, labels).

    images: (N, 32, 32, 3) uint8, RGB
    labels: (N,) int
    """

    batch = unpickle(batch_path)

    raw = np.asarray(batch[b"data"], dtype=np.uint8)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)

    images = raw.reshape(-1, *CIFAR_IMAGE_SHAPE)
    images = images.transpose(0, 2, 3, 1)  # -> (N, H, W, C), RGB

    return images, labels


def load_cifar_batches(batch_paths):
    all_images = []
    all_labels = []

    for batch_path in batch_paths:
        images, labels = load_cifar_batch(batch_path)
        all_images.append(images)
        all_labels.append(labels)

    return np.concatenate(all_images), np.concatenate(all_labels)


def subsample_per_class(images, labels, samples_per_class):
    rng = np.random.default_rng(RANDOM_STATE)

    selected_indices = []

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)

        if len(class_indices) > samples_per_class:
            class_indices = rng.choice(
                class_indices,
                size=samples_per_class,
                replace=False
            )

        selected_indices.append(class_indices)

    selected_indices = np.concatenate(selected_indices)
    rng.shuffle(selected_indices)

    return images[selected_indices], labels[selected_indices]


# ---------- Feature extraction ----------

def extract_features(image_rgb):
    """Convert one RGB image array into a fixed-length HOG feature vector."""

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(
        gray,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Fixed pixel normalization: [0, 255] -> [0, 1]
    gray = gray.astype(np.float32) / 255.0

    features = hog(
        gray,
        feature_vector=True,
        channel_axis=None,
        **HOG_PARAMS
    )

    return features.astype(np.float32)


def build_feature_matrix(images):
    features = [extract_features(image) for image in images]
    return np.stack(features)


# ---------- Five-fold cross-validation and model selection ----------

def train_model(X, y):
    """Compare every (K, metric) pair by five-fold CV, refit the winner.

    Returns (best_estimator, selection_summary). The summary carries the
    full comparison table so it can be written to the results file.
    """

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Reuse exactly the same folds for every parameter setting.
    folds = list(cv.split(X, y))

    # Avoid choosing K larger than a fold's training subset.
    smallest_training_fold = min(
        len(train_indices)
        for train_indices, _ in folds
    )

    candidate_k = [
        k for k in CANDIDATE_K
        if k <= smallest_training_fold
    ]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            algorithm="brute",
            weights="uniform",
            n_jobs=N_JOBS
        ))
    ])

    parameter_grid = {
        "knn__n_neighbors": candidate_k,
        "knn__metric": CANDIDATE_METRICS
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="accuracy",
        cv=folds,
        refit=True,
        n_jobs=N_JOBS,
        error_score="raise"
    )

    print(
        f"\nSearching {len(candidate_k)} K values x "
        f"{len(CANDIDATE_METRICS)} metrics "
        f"({len(candidate_k) * len(CANDIDATE_METRICS) * N_FOLDS} fits)..."
    )

    start_time = time.perf_counter()
    search.fit(X, y)
    search_seconds = time.perf_counter() - start_time

    rows = collect_cv_rows(search)

    print("\nCross-validation results:")

    for row in sorted(rows, key=lambda r: (r["metric"], r["k"])):
        print(
            f"K={row['k']:2d}, "
            f"metric={row['metric']:9s}: "
            f"{row['mean']:.2%} +/- {row['std']:.2%}"
        )

    print("\nSelected parameters:", search.best_params_)
    print(f"Best mean validation accuracy: {search.best_score_:.2%}")
    print(f"Grid search took {search_seconds:.1f}s")

    best_row = next(row for row in rows if row["is_best"])

    print("\nValidation scores for the selected configuration:")

    for fold_number, score in enumerate(best_row["fold_scores"], start=1):
        print(f"Fold {fold_number}: {score:.2%}")

    summary = {
        "rows": rows,
        "best": best_row,
        "candidate_k": candidate_k,
        "search_seconds": search_seconds,
    }

    # GridSearchCV has already refitted this pipeline
    # on all training data because refit=True.
    return search.best_estimator_, summary


def collect_cv_rows(search):
    """Flatten GridSearchCV results into one record per (K, metric) pair."""

    results = search.cv_results_

    rows = []

    for index, params in enumerate(results["params"]):
        rows.append({
            "k": params["knn__n_neighbors"],
            "metric": params["knn__metric"],
            "mean": results["mean_test_score"][index],
            "std": results["std_test_score"][index],
            "rank": int(results["rank_test_score"][index]),
            "fit_seconds": results["mean_fit_time"][index],
            "score_seconds": results["mean_score_time"][index],
            "fold_scores": [
                results[f"split{fold}_test_score"][index]
                for fold in range(N_FOLDS)
            ],
            "is_best": index == search.best_index_,
        })

    return rows


# ---------- Evaluation on the held-out CIFAR-10 test batch ----------

def evaluate_model(model, X_test, y_test, label_names):
    predictions = model.predict(X_test)

    scores = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
    }

    per_class = classification_report(
        y_test,
        predictions,
        target_names=label_names,
        zero_division=0,
        output_dict=True
    )

    matrix = confusion_matrix(y_test, predictions)

    print("\n===== Evaluation on test_batch =====")
    print(f"Accuracy:  {scores['accuracy']:.2%}")
    print(f"Precision: {scores['precision']:.2%} (macro)")
    print(f"Recall:    {scores['recall']:.2%} (macro)")
    print(f"F1-score:  {scores['f1']:.2%} (macro)")

    print("\nPer-class report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=label_names,
            zero_division=0
        )
    )

    print("Confusion matrix (rows=true, cols=predicted):")
    print(matrix)

    return {
        "scores": scores,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


# ---------- Markdown report ----------

def markdown_table(header, rows):
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]

    lines.extend(
        "| " + " | ".join(cells) + " |"
        for cells in rows
    )

    return lines


def k_comparison_section(summary):
    """One table per metric, so the effect of K alone is easy to read."""

    lines = []

    for metric in CANDIDATE_METRICS:
        metric_rows = sorted(
            (row for row in summary["rows"] if row["metric"] == metric),
            key=lambda row: row["k"]
        )

        if not metric_rows:
            continue

        lines.append(f"### Metric: `{metric}`")
        lines.append("")

        lines.extend(markdown_table(
            ["K", "Mean accuracy", "Std dev", "Rank"] +
            [f"Fold {i}" for i in range(1, N_FOLDS + 1)],
            [
                [
                    str(row["k"]),
                    f"{row['mean']:.2%}" + (" **(best)**" if row["is_best"] else ""),
                    f"{row['std']:.2%}",
                    str(row["rank"]),
                ] + [f"{score:.2%}" for score in row["fold_scores"]]
                for row in metric_rows
            ]
        ))

        best_for_metric = max(metric_rows, key=lambda row: row["mean"])
        lines.append("")
        lines.append(
            f"Best K for `{metric}`: **{best_for_metric['k']}** "
            f"at {best_for_metric['mean']:.2%} mean validation accuracy."
        )
        lines.append("")

    return lines


def confusion_matrix_section(matrix, label_names):
    lines = ["Rows are the true class, columns the predicted class.", ""]

    lines.extend(markdown_table(
        ["true \\ pred"] + label_names,
        [
            [label_names[row_index]] + [str(int(count)) for count in row]
            for row_index, row in enumerate(matrix)
        ]
    ))

    return lines


def write_results_markdown(
    summary, evaluation, label_names, dataset_info, feature_dimensions
):
    best = summary["best"]
    scores = evaluation["scores"]
    per_class = evaluation["per_class"]

    lines = [
        "# KNN on CIFAR-10 with HOG features",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by `{Path(__file__).name}`.",
        "",
        "## Run configuration",
        "",
    ]

    lines.extend(markdown_table(
        ["Setting", "Value"],
        [
            ["Training images", f"{dataset_info['n_train']} "
                               f"({TRAIN_SAMPLES_PER_CLASS}/class)"],
            ["Test images", f"{dataset_info['n_test']} "
                            f"({TEST_SAMPLES_PER_CLASS}/class)"],
            ["Classes", str(len(label_names))],
            ["Feature dimensions", str(feature_dimensions)],
            ["Preprocessing", "StandardScaler fitted inside each CV fold"],
            ["Cross-validation", f"Stratified {N_FOLDS}-fold, shuffled"],
            ["K values tested", ", ".join(str(k) for k in summary["candidate_k"])],
            ["Distance metrics tested", ", ".join(CANDIDATE_METRICS)],
            ["Grid search time", f"{summary['search_seconds']:.1f}s"],
        ]
    ))

    lines += [
        "",
        "## Choosing K",
        "",
        "Every configuration below was scored on the *same* five folds, so the "
        "numbers are directly comparable. The model reported in the next "
        "section is the winner refitted on the full training subsample.",
        "",
    ]

    lines.extend(k_comparison_section(summary))

    lines += [
        "## Selected model",
        "",
    ]

    lines.extend(markdown_table(
        ["Property", "Value"],
        [
            ["K (n_neighbors)", str(best["k"])],
            ["Distance metric", best["metric"]],
            ["Weights", "uniform"],
            ["Mean validation accuracy", f"{best['mean']:.2%}"],
            ["Validation std dev", f"{best['std']:.2%}"],
            ["Per-fold validation accuracy",
             ", ".join(f"{score:.2%}" for score in best["fold_scores"])],
        ]
    ))

    lines += [
        "",
        "## Test-set performance (`test_batch`)",
        "",
    ]

    lines.extend(markdown_table(
        ["Metric", "Value"],
        [
            ["Accuracy", f"{scores['accuracy']:.2%}"],
            ["Precision (macro)", f"{scores['precision']:.2%}"],
            ["Recall (macro)", f"{scores['recall']:.2%}"],
            ["F1-score (macro)", f"{scores['f1']:.2%}"],
        ]
    ))

    lines += [
        "",
        "### Per-class results",
        "",
    ]

    lines.extend(markdown_table(
        ["Class", "Precision", "Recall", "F1-score", "Support"],
        [
            [
                name,
                f"{per_class[name]['precision']:.2%}",
                f"{per_class[name]['recall']:.2%}",
                f"{per_class[name]['f1-score']:.2%}",
                str(int(per_class[name]["support"])),
            ]
            for name in label_names
        ]
    ))

    lines += [
        "",
        "### Confusion matrix",
        "",
    ]

    lines.extend(
        confusion_matrix_section(evaluation["confusion_matrix"], label_names)
    )

    lines.append("")

    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nResults written to {RESULTS_FILE}")


# ---------- Main program ----------

def main():
    label_names = load_label_names(CIFAR_DIR / META_FILE)

    train_batch_paths = [
        CIFAR_DIR / name for name in TRAIN_BATCH_FILES
    ]
    train_images, train_labels = load_cifar_batches(train_batch_paths)
    train_images, train_labels = subsample_per_class(
        train_images, train_labels, TRAIN_SAMPLES_PER_CLASS
    )

    test_images, test_labels = load_cifar_batch(
        CIFAR_DIR / TEST_BATCH_FILE
    )
    test_images, test_labels = subsample_per_class(
        test_images, test_labels, TEST_SAMPLES_PER_CLASS
    )

    print("Training images:", train_images.shape[0])
    print(
        "Training images per class:",
        Counter(train_labels.tolist())
    )
    print("Test images:", test_images.shape[0])

    X_train = build_feature_matrix(train_images)
    y_train = train_labels

    print("\nTraining feature matrix:", X_train.shape)

    model, summary = train_model(X_train, y_train)

    X_test = build_feature_matrix(test_images)
    y_test = test_labels

    evaluation = evaluate_model(model, X_test, y_test, label_names)

    write_results_markdown(
        summary,
        evaluation,
        label_names,
        dataset_info={
            "n_train": int(train_images.shape[0]),
            "n_test": int(test_images.shape[0]),
        },
        feature_dimensions=int(X_train.shape[1]),
    )


if __name__ == "__main__":
    main()
