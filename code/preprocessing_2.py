from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple
import re

WORD = 0
TAG = 1

# Define a small lexicon of common measurement unit tokens (lowercase)
UNIT_WORDS = {"mg","g","kg","ug","\u03bcg","ml","l","\u03bcl","mm","cm","nm","um","hr","h","min","bp"}

BIOMED_SUFFIXES = ["ase", "itis", "emia", "oma", "cyte", "genic", "phage", "troph", "blast", "scope", "lysis"]

def normalize_word(word: str) -> str:
    return word.lower()

def word_shape(word: str) -> str:
    shape = ""
    for c in word:
        if c.isupper(): shape += "X"
        elif c.islower(): shape += "x"
        elif c.isdigit(): shape += "d"
        else: shape += c
    return shape

def word_short_shape(word: str) -> str:
    long_shape = word_shape(word)
    if not long_shape:
        return ""
    short_shape = long_shape[0]
    for char in long_shape[1:]:
        if char != short_shape[-1]:
            short_shape += char
    return short_shape

class FeatureStatistics:
    def __init__(self):
        self.tags = {"~"}
        self.histories = []
        feature_dict_list = [
            "f100", "f101", "f102", "f103", "f104",
            "f106", "f107", "f110", "f111", "f112",
            "f113", "f115", "f116", "f117", "f118", "f119"
        ]
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}

    def get_word_tag_pair_count(self, file_path) -> None:
        with open(file_path) as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    norm_word = normalize_word(cur_word)
                    self.tags.add(cur_tag)

                    self.feature_rep_dict["f100"][(norm_word, cur_tag)] = \
                        self.feature_rep_dict["f100"].get((norm_word, cur_tag), 0) + 1

                    prev_tag = split_words[word_idx - 1].rsplit('_', 1)[1] if word_idx > 0 else "*"
                    self.feature_rep_dict["f101"][(prev_tag, cur_tag)] = \
                        self.feature_rep_dict["f101"].get((prev_tag, cur_tag), 0) + 1

                    for l in [1, 2, 3]:
                        if len(cur_word) >= l:
                            suf = cur_word[-l:].lower()
                            pre = cur_word[:l].lower()
                            self.feature_rep_dict["f102"][(suf, cur_tag)] = \
                                self.feature_rep_dict["f102"].get((suf, cur_tag), 0) + 1
                            self.feature_rep_dict["f103"][(pre, cur_tag)] = \
                                self.feature_rep_dict["f103"].get((pre, cur_tag), 0) + 1

                    if word_idx > 1:
                        prev_prev_tag = split_words[word_idx - 2].rsplit('_', 1)[1]
                    else:
                        prev_prev_tag = "*"
                    self.feature_rep_dict["f104"][(prev_prev_tag, prev_tag, cur_tag)] = \
                        self.feature_rep_dict["f104"].get((prev_prev_tag, prev_tag, cur_tag), 0) + 1

                    prev_word = normalize_word(split_words[word_idx - 1].rsplit('_', 1)[0]) if word_idx > 0 else "*"
                    if prev_word != "*" and not re.fullmatch(r"[^A-Za-z0-9]+", prev_word):
                        self.feature_rep_dict["f106"][(prev_word, cur_tag)] = \
                            self.feature_rep_dict["f106"].get((prev_word, cur_tag), 0) + 1

                    next_word = normalize_word(split_words[word_idx + 1].rsplit('_', 1)[0]) \
                        if word_idx < len(split_words) - 1 else "~"
                    if next_word != "~" and not re.fullmatch(r"[^A-Za-z0-9]+", next_word):
                        self.feature_rep_dict["f107"][(next_word, cur_tag)] = \
                            self.feature_rep_dict["f107"].get((next_word, cur_tag), 0) + 1

                    if '.' in cur_word:
                        self.feature_rep_dict["f110"][('.', cur_tag)] = \
                            self.feature_rep_dict["f110"].get(('.', cur_tag), 0) + 1

                    if '-' in cur_word:
                        self.feature_rep_dict["f111"][('-', cur_tag)] = \
                            self.feature_rep_dict["f111"].get(('-', cur_tag), 0) + 1

                    self.feature_rep_dict["f112"][(str(len(cur_word)), cur_tag)] = \
                        self.feature_rep_dict["f112"].get((str(len(cur_word)), cur_tag), 0) + 1

                    if norm_word in UNIT_WORDS:
                        self.feature_rep_dict["f113"][("UNIT", cur_tag)] = \
                            self.feature_rep_dict["f113"].get(("UNIT", cur_tag), 0) + 1

                    short_shape = word_short_shape(cur_word)
                    if short_shape:
                        self.feature_rep_dict["f115"][(short_shape, cur_tag)] = \
                            self.feature_rep_dict["f115"].get((short_shape, cur_tag), 0) + 1

                    if re.fullmatch(r"[A-Z]{2,}", cur_word):
                        self.feature_rep_dict["f116"][("ACRONYM", cur_tag)] = \
                            self.feature_rep_dict["f116"].get(("ACRONYM", cur_tag), 0) + 1

                    if re.fullmatch(r"\d+(mg|ml|%)", norm_word):
                        self.feature_rep_dict["f117"][("NUM_UNIT", cur_tag)] = \
                            self.feature_rep_dict["f117"].get(("NUM_UNIT", cur_tag), 0) + 1

                    for suffix in BIOMED_SUFFIXES:
                        if norm_word.endswith(suffix):
                            self.feature_rep_dict["f118"][(suffix, cur_tag)] = \
                                self.feature_rep_dict["f118"].get((suffix, cur_tag), 0) + 1

                    if cur_word[0].isupper() and cur_word[1:].islower():
                        self.feature_rep_dict["f119"][("INITCAP", cur_tag)] = \
                            self.feature_rep_dict["f119"].get(("INITCAP", cur_tag), 0) + 1

                sentence = [("", ""), ("", "")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("", ""))
                for i in range(2, len(sentence) - 1):
                    history = (
                        sentence[i][0], sentence[i][1],
                        sentence[i - 1][0], sentence[i - 1][1],
                        sentence[i - 2][0], sentence[i - 2][1],
                        sentence[i + 1][0]
                    )
                    self.histories.append(history)

    def compute_feature_scores(self):
        scores = {}
        for feat_type, features in self.feature_rep_dict.items():
            total = sum(features.values())
            if total == 0:
                continue
            scores[feat_type] = {feat: count / total for feat, count in features.items()}
        return scores

    def filter_top_scoring_features(self, top_percentile=0.6, min_threshold=5):
        scores = self.compute_feature_scores()
        for feat_type, score_dict in scores.items():
            sorted_feats = sorted(score_dict.items(), key=lambda x: -x[1])
            cutoff = int(len(sorted_feats) * top_percentile)
            top_feats = set()
            for feat, _ in sorted_feats[:cutoff]:
                count = self.feature_rep_dict[feat_type].get(feat, 0)
                if count >= min_threshold:
                    top_feats.add(feat)
            self.feature_rep_dict[feat_type] = OrderedDict(
                (f, self.feature_rep_dict[feat_type][f])
                for f in self.feature_rep_dict[feat_type] if f in top_feats
            )

class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        self.feature_statistics = feature_statistics
        self.threshold = threshold
        self.n_total_features = 0
        # Initialize feature-to-index mapping for each feature class
        self.feature_to_idx = {feat: OrderedDict() for feat in feature_statistics.feature_rep_dict}
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        """Assigns an index to each feature that appears >= threshold times."""
        for feat_class, feats in self.feature_statistics.feature_rep_dict.items():
            for feat, count in feats.items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        # Build sparse feature matrices for training (small_matrix) and all histories (big_matrix)
        big_r = 0
        big_rows, big_cols = [], []
        small_rows, small_cols = [], []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r); small_cols.append(c)
            for y_tag in self.feature_statistics.tags:
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r); big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        # Create sparse matrices
        self.big_matrix = sparse.csr_matrix(
            (np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
            shape=(len(self.feature_statistics.tags) * len(self.feature_statistics.histories), self.n_total_features),
            dtype=bool
        )
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(self.feature_statistics.histories), self.n_total_features),
            dtype=bool
        )

def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple, int]]) -> List[int]:
    c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word = history
    features = []

    def add_feature(feat_class, key):
        if key in dict_of_dicts.get(feat_class, {}):
            features.append(dict_of_dicts[feat_class][key])

    # f100: current word (lowercased)
    add_feature("f100", (c_word.lower(), c_tag))
    # f101: previous tag
    add_feature("f101", (p_tag, c_tag))

    # f102/f103: suffix and prefix of length 1,2,3
    for l in [1, 2, 3]:
        if len(c_word) >= l:
            add_feature("f102", (c_word[-l:].lower(), c_tag))
            add_feature("f103", (c_word[:l].lower(), c_tag))

    # f104: tag trigram
    add_feature("f104", (pp_tag, p_tag, c_tag))

    # f106: previous word (lowercased)
    if p_word != "*" and not re.fullmatch(r"[^A-Za-z0-9]+", p_word):
        add_feature("f106", (p_word.lower(), c_tag))
    # f107: next word (lowercased)
    if n_word != "~" and not re.fullmatch(r"[^A-Za-z0-9]+", n_word):
        add_feature("f107", (n_word.lower(), c_tag))

    # f110: contains '.'
    if '.' in c_word:
        add_feature("f110", (".", c_tag))
    # f111: contains '-'
    if '-' in c_word:
        add_feature("f111", ("-", c_tag))

    # f112: word length
    add_feature("f112", (str(len(c_word)), c_tag))
    # f113: unit indicator
    if c_word.lower() in UNIT_WORDS:
        add_feature("f113", ("UNIT", c_tag))

    # f115: short word shape
    add_feature("f115", (word_short_shape(c_word), c_tag))
    # f116: acronym (all caps)
    if re.fullmatch(r"[A-Z]{2,}", c_word):
        add_feature("f116", ("ACRONYM", c_tag))
    # f117: numeric unit pattern
    if re.fullmatch(r"\d+(mg|ml|%)", c_word.lower()):
        add_feature("f117", ("NUM_UNIT", c_tag))
    # f118: specific suffix
    for suffix in ["ase", "itis", "emia", "oma", "cyte", "genic"]:
        if c_word.lower().endswith(suffix):
            add_feature("f118", (suffix, c_tag))

    # f119: InitCap (first letter uppercase, rest lowercase)
    if c_word and c_word[0].isupper() and c_word[1:].islower():
        add_feature("f119", ("INITCAP", c_tag))

    return features

def preprocess_train(train_path, threshold):
    stats = FeatureStatistics()
    stats.get_word_tag_pair_count(train_path)
    # Optionally filter out low-scoring features to further reduce dimensionality
    stats.filter_top_scoring_features(top_percentile=0.6, min_threshold=2)
    feat2id = Feature2id(stats, threshold)
    feat2id.get_features_idx()
        # 🔍 Print number of features per class
    for feat_class, mapping in feat2id.feature_to_idx.items():
        print(f"{feat_class}: {len(mapping)} features")
    feat2id.calc_represent_input_with_features()
    return stats, feat2id

def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    Reads a test file (tagged or untagged) and returns a list of sentences.
    Each sentence is represented as a tuple: ([words], [tags]).
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["", ""], ["", ""])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].rsplit('_', 1)
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences