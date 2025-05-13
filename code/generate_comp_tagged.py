"""
generate_comp_tagged.py

This script loads a trained MEMM model and uses it to tag a competition dataset
(e.g., comp1.words or comp2.words). It outputs a tagged .wtag file with the format:
word_TAG for each token in the input.

Usage:
    python3 code/generate_comp_tagged.py --version 1
    python3 code/generate_comp_tagged.py --version 2

Requirements:
- The model must already be trained and the weights saved as:
    trained_models/weights_1.pkl or weights_2.pkl
    the models are saved under the folder "trained_mededls"
- The corresponding competition file must exist at:
    data/comp1.words or data/comp2.words under the folder "data"

Output:
- The output tagged file will be saved as:
    comp1_generate_predict.wtag or comp2_generate_predict.wtag (in the project root)

Arguments:
--version [1|2] : Choose the preprocessing version used during training.
"""

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