## HMHH_NLP

This project implements a MEMM-based POS tagger using different preprocessing feature sets.

### ✅ Setup

Install all required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 🚀 Running the Tagger

Run the tagger with a specified preprocessing version:

```bash
python3 main.py --version 1  # uses preprocessing_1
python3 main.py --version 2  # uses preprocessing_2
```

The script will:
- Preprocess the training data
- Train or load weights from `weights_<version>.pkl`
- Run inference on the corresponding test set
- Save predictions to `predictions_<version>.wtag`
- Evaluate and display a confusion matrix

### ✨ Tag the comp set using a trained model

Run this from the **root** of the repository (not from the `code/` folder):

```bash
python code/generate_comp_tagged.py --version 1
```

This will:
- Load the trained model from `trained_models/weights_1.pkl`
- Read the input from `data/comp1.words`
- Generate tagged output at `output/comp1_predict.wtag`

Make sure you have trained the model first by running:

```bash
python main.py --version 1
```

(Use `--version 2` for the alternate preprocessing model.)


### 🧰 Utility Scripts

Strip tags from a `.wtag` file (remove all POS tags and underscores):

```bash
python strip_and_compare_wtag.py --strip input.wtag output.txt
```

Compare two `.wtag` files and highlight mismatched predictions:

```bash
python strip_and_compare_wtag.py --compare predictions1.wtag predictions2.wtag
```
