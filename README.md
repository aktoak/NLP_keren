## HMHH_NLP

This project implements a MEMM-based POS tagger using different preprocessing feature sets.

### ✅ Setup

Install all required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 🚀 Running the Model tranining to generate weights and predictions

Run the tagger with a specified preprocessing version:

```bash
python3 main.py --version 1  # uses preprocessing_1 and train1
python3 main.py --version 2 --kfold # uses preprocessing_2 and train2
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
# generateds comp1_generate_predict.wtag against data/comp1.words
python code/generate_comp_tagged.py --version 1 

# generateds comp2_generate_predict.wtag against data/compo2.words
python code/generate_comp_tagged.py --version 2 
```

This will:
- Load the trained model from `trained_models/weights_1.pkl`
- Read the input from `data/comp1.words`
- Generate tagged output at `output/comp1_predict.wtag`

Make sure you have trained the model first by running:

```bash
python code/main.py --version 1

# for train2 data
python3 main.py --version 2 --kfold
```