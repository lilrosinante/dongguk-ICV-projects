from pathlib import Path
from collections import Counter
import pickle

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

TRAIN_BATCH_FILES = [
    f"data_batch_{i}" for i in range(1, 6)
]
TEST_BATCH_FILE = "test_batch"
META_FILE = "batches.meta"

CIFAR_IMAGE_SHAPE = (3, 32, 32)  # channels, height, width as stored

IMAGE_SIZE = (64, 64)  # OpenCV uses (width, height)
N_FOLDS = 5
RANDOM_STATE = 42

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
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
        channel_axis=None
    )

    return features.astype(np.float32)


def build_feature_matrix(images):
    features = [extract_features(image) for image in images]
    return np.stack(features)


# ---------- Five-fold cross-validation and model selection ----------

def train_model(X, y):
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
        k for k in [1, 3, 5, 7, 9]
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
        "knn__metric": ["manhattan", "euclidean"]
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

    search.fit(X, y)

    print("\nCross-validation results:")

    for params, mean, std in zip(
        search.cv_results_["params"],
        search.cv_results_["mean_test_score"],
        search.cv_results_["std_test_score"]
    ):
        print(
            f"K={params['knn__n_neighbors']:2d}, "
            f"metric={params['knn__metric']:9s}: "
            f"{mean:.2%} ± {std:.2%}"
        )

    print("\nSelected parameters:", search.best_params_)
    print(f"Best mean validation accuracy: {search.best_score_:.2%}")

    best_index = search.best_index_

    print("\nValidation scores for the selected configuration:")

    for fold_number in range(N_FOLDS):
        score = search.cv_results_[
            f"split{fold_number}_test_score"
        ][best_index]

        print(f"Fold {fold_number + 1}: {score:.2%}")

    # GridSearchCV has already refitted this pipeline
    # on all training data because refit=True.
    return search.best_estimator_


# ---------- Evaluation on the held-out CIFAR-10 test batch ----------

def evaluate_model(model, X_test, y_test, label_names):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    print("\n===== Evaluation on test_batch =====")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%} (macro)")
    print(f"Recall:    {recall:.2%} (macro)")
    print(f"F1-score:  {f1:.2%} (macro)")

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
    print(confusion_matrix(y_test, predictions))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


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

    model = train_model(X_train, y_train)

    X_test = build_feature_matrix(test_images)
    y_test = test_labels

    evaluate_model(model, X_test, y_test, label_names)


if __name__ == "__main__":
    main()
