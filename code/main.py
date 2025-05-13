"""
Main script for training and evaluating a sequence labeling model using structured features.

Features:
- Supports multiple preprocessing versions.
- Optionally runs k-fold cross-validation to find the best regularization parameter (lambda).
- Trains the model and runs inference.
- Evaluates accuracy (only for version 1, which has a labeled test set).
- Always generates a prediction file for `comp{version}.words`:
    ➤ comp1_208152439_203912506.wtag
    ➤ comp2_208152439_203912506.wtag

Dependencies:
- numpy, matplotlib, scikit-learn, tqdm
- Local modules: `inference.py`, `optimization.py`, and versioned `preprocessing_x.py`
"""

import importlib
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from tqdm import tqdm

from optimization import get_optimal_vector
from inference import tag_all_test


def load_preprocessing(version):
    """
    Dynamically loads the preprocessing module for the specified version.
    Assumes modules named preprocessing_1.py, preprocessing_2.py, etc.

    Returns:
        preprocess_train: function to extract features and train
        read_test: function to read test input
        represent_input_with_features: feature representation function
    """
        
    module = importlib.import_module(f"preprocessing_{version}")
    return module.preprocess_train, module.read_test, module.represent_input_with_features


def evaluate_accuracy(pred_path, gold_path):
    correct = total = 0
    y_true, y_pred = [], []

    with open(pred_path) as pred_f, open(gold_path) as gold_f:
        for i, (pred_line, gold_line) in enumerate(zip(pred_f, gold_f)):
            pred_tags = [w.split('_')[-1] for w in pred_line.strip().split()]
            gold_tags = [w.split('_')[-1] for w in gold_line.strip().split()]
            assert len(pred_tags) == len(gold_tags), f"Mismatch in line {i}"

            for p, g in zip(pred_tags, gold_tags):
                y_pred.append(p)
                y_true.append(g)
                total += 1
                if p == g:
                    correct += 1

    accuracy = correct / total if total else 0
    print(f"\n🔍 Accuracy: {accuracy:.4f} ({correct}/{total} correct tags)")

    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation='vertical', cmap='Blues')
    plt.tight_layout()
    plt.title("Confusion Matrix")
    plt.show()

    return accuracy


def run_kfold_validation(train_path, k=5, version=1, threshold=1, lambda_values=None):
    """
    Performs k-fold cross-validation to select the best lambda.

    Args:
        train_path (str): Path to training data
        k (int): Number of folds
        version (int): Preprocessing version
        threshold (int): Minimum feature occurrence
        lambda_values (List[float]): List of lambdas to test

    Returns:
        float: Lambda value with best average accuracy
    """
    if lambda_values is None:
        lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    print(f"📂 Reading training data from {train_path}...")
    with open(train_path, 'r') as f:
        all_sentences = [line.strip() for line in f if line.strip()]

    print(f"🔄 Running {k}-fold cross validation with preprocessing version {version}...")

    os.makedirs("kfold_temp", exist_ok=True)
    preprocess_train, read_test, represent_input_with_features = load_preprocessing(version)
    all_accuracies = {lam: [] for lam in lambda_values}
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    best_lambda = None
    best_avg_accuracy = 0

    for lam in lambda_values:
        print(f"\n🔍 Testing lambda={lam}")
        fold_accuracies = []

        for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(all_sentences), total=k)):
            train_data = [all_sentences[i] for i in train_idx]
            test_data = [all_sentences[i] for i in test_idx]

            temp_train_path = os.path.join("kfold_temp", f"fold{fold}_train.wtag")
            temp_test_path = os.path.join("kfold_temp", f"fold{fold}_test.wtag")
            temp_weights_path = os.path.join("kfold_temp", f"fold{fold}_weights.pkl")
            temp_pred_path = os.path.join("kfold_temp", f"fold{fold}_pred.wtag")

            with open(temp_train_path, 'w') as f:
                f.write('\n'.join(train_data))
            with open(temp_test_path, 'w') as f:
                f.write('\n'.join(test_data))

            statistics, feature2id = preprocess_train(temp_train_path, threshold)
            get_optimal_vector(statistics, feature2id, lam=lam, weights_path=temp_weights_path)

            with open(temp_weights_path, 'rb') as f:
                optimal_params, feature2id = pickle.load(f)
            pre_trained_weights = optimal_params[0]

            tag_all_test(temp_test_path, pre_trained_weights, feature2id, temp_pred_path,
                         read_test, represent_input_with_features)

            with open(temp_pred_path) as pred_f, open(temp_test_path) as gold_f:
                correct = total = 0
                for pred_line, gold_line in zip(pred_f, gold_f):
                    pred_tags = [w.split('_')[-1] for w in pred_line.strip().split()]
                    gold_tags = [w.split('_')[-1] for w in gold_line.strip().split()]
                    for p, g in zip(pred_tags, gold_tags):
                        total += 1
                        if p == g:
                            correct += 1

                accuracy = correct / total if total else 0
                fold_accuracies.append(accuracy)

        avg_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        all_accuracies[lam] = fold_accuracies

        print(f"Lambda={lam}: Avg accuracy={avg_accuracy:.4f} (±{std_accuracy:.4f})")

        if avg_accuracy > best_avg_accuracy:
            best_avg_accuracy = avg_accuracy
            best_lambda = lam

    for file in os.listdir("kfold_temp"):
        os.remove(os.path.join("kfold_temp", file))
    os.rmdir("kfold_temp")

    plt.figure(figsize=(10, 6))
    for lam, accuracies in all_accuracies.items():
        plt.plot([i + 1 for i in range(k)], accuracies, 'o-', label=f'λ={lam}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title(f'{k}-Fold Cross Validation Results (Preprocessing v{version})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.figure(figsize=(8, 5))
    avg_accuracies = [np.mean(all_accuracies[lam]) for lam in lambda_values]
    plt.plot(lambda_values, avg_accuracies, 'o-', linewidth=2)
    plt.xlabel('Lambda Value')
    plt.ylabel('Average Accuracy')
    plt.title('Effect of Regularization Parameter on Model Performance')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'kfold_results_v{version}.png')
    plt.show()

    print(f"\n✅ Best lambda value: {best_lambda} with average accuracy: {best_avg_accuracy:.4f}")
    return best_lambda


def main(version=1, use_kfold=False, k=5, lambda_values=None):
    """
    Main entry point:
    - Optionally runs k-fold to find lambda
    - Trains the model
    - Runs inference and evaluation as needed
    - Always produces a comp{version}_208152439_203912506.wtag file
    """
    threshold = 1
    lam = 0.1

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_path = os.path.join(ROOT_DIR, "data", f"train{version}.wtag")

    if version == 1:
        test_path = os.path.join(ROOT_DIR, "data", f"test{version}.wtag")
    else:
        test_path = None

    weights_path = os.path.join(ROOT_DIR, "trained_models", f"weights_{version}.pkl")
    predictions_path = os.path.join(ROOT_DIR, f"predict{version}_208152439.wtag")

    preprocess_train, read_test, represent_input_with_features = load_preprocessing(version)

    if use_kfold:
        print(f"🔄 Running {k}-fold cross validation...")
        lam = run_kfold_validation(train_path, k, version, threshold, lambda_values)
        print(f"🎯 Using optimal lambda: {lam}")

        print("🔧 Preprocessing full training data...")
        statistics, feature2id = preprocess_train(train_path, threshold)

        print("📉 Training full model with optimal lambda...")
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        get_optimal_vector(statistics, feature2id, weights_path=weights_path, lam=lam)

        with open(weights_path, 'rb') as f:
            optimal_params, feature2id = pickle.load(f)
        pre_trained_weights = optimal_params[0]

        print(f"✅ Norm: {np.linalg.norm(pre_trained_weights):.4f}")
    else:
        if os.path.exists(weights_path):
            print("📦 Loading existing weights...")
            with open(weights_path, 'rb') as f:
                optimal_params, feature2id = pickle.load(f)
            pre_trained_weights = optimal_params[0]
        else:
            print("🔧 Preprocessing...")
            statistics, feature2id = preprocess_train(train_path, threshold)

            print("📉 Training...")
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            get_optimal_vector(statistics, feature2id, weights_path=weights_path, lam=lam)

            with open(weights_path, 'rb') as f:
                optimal_params, feature2id = pickle.load(f)
            pre_trained_weights = optimal_params[0]

        print(f"✅ Norm: {np.linalg.norm(pre_trained_weights):.4f}")

    # === Inference ===
    comp_words_path = os.path.join(ROOT_DIR, "data", f"comp{version}.words")
    comp_output_path = os.path.join(ROOT_DIR, f"comp{version}_m{version}_208152439_203912506.wtag")

    if version == 1:
        if os.path.exists(predictions_path):
            print("🧹 Removing old test predictions...")
            os.remove(predictions_path)

        print("📤 Running inference on test set...")
        tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path,
                     read_test, represent_input_with_features)

        print("📊 Evaluating accuracy on test set...")
        evaluate_accuracy(predictions_path, test_path)

        print("📤 Running inference on comp1.words...")
        if os.path.exists(comp_output_path):
            os.remove(comp_output_path)

        tag_all_test(comp_words_path, pre_trained_weights, feature2id, comp_output_path,
                     read_test, represent_input_with_features)

        print(f"✅ Predictions written to: {comp_output_path}")

    elif version == 2:
        print("📤 Running inference on comp2.words...")
        if os.path.exists(comp_output_path):
            os.remove(comp_output_path)

        tag_all_test(comp_words_path, pre_trained_weights, feature2id, comp_output_path,
                     read_test, represent_input_with_features)

        print(f"✅ Predictions written to: {comp_output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1, choices=[1, 2], help="Preprocessing version")
    parser.add_argument("--kfold", action="store_true", help="Use k-fold cross validation")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for cross validation")
    parser.add_argument("--lambda_values", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0, 5.0],
                        help="Lambda values to test during cross validation")

    args = parser.parse_args()

    main(version=args.version, use_kfold=args.kfold, k=args.k, lambda_values=args.lambda_values)
