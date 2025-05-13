import importlib
import os
import pickle
import numpy as np
import sys

# Allow importing from root if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimization import get_optimal_vector
from inference import tag_all_test

def load_preprocessing(version):
    module = importlib.import_module(f"preprocessing_{version}")
    return module.preprocess_train, module.read_test, module.represent_input_with_features

def main(version=1):
    # Paths relative to the root directory
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    weights_path = os.path.join(ROOT_DIR, "trained_models", f"weights_{version}.pkl")
    predictions_path = os.path.join(ROOT_DIR, f"comp{version}_generate_predict.wtag")
    comp_words_path = os.path.join(ROOT_DIR, f"data",f"comp{version}.words")


    _, read_test, represent_input_with_features = load_preprocessing(version)

    if not os.path.exists(comp_words_path):
        raise FileNotFoundError(f"❌ Input file '{comp_words_path}' not found. Please make sure it exists.")

    if os.path.exists(predictions_path):
        print("🧹 Removing old predictions...")
        os.remove(predictions_path)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"❌ Weights file '{weights_path}' not found.\n"
            f"Please train the model first using:\n\n"
            f"    python main.py --version {version}"
        )

    print("📦 Loading weights...")
    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(f"✅ Weight vector norm: {np.linalg.norm(pre_trained_weights):.4f}")

    print("📤 Running inference...")
    tag_all_test(comp_words_path, pre_trained_weights, feature2id, predictions_path,
                 read_test, represent_input_with_features)

    print(f"✅ Predictions written to: {predictions_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1, choices=[1, 2], help="Preprocessing version")
    args = parser.parse_args()

    main(version=args.version)