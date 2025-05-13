from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple

WORD = 0
TAG = 1


class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        feature_dict_list = [
            "f100",  # (word, tag)
            "f101",  # (prev_tag, cur_tag)
            "f102",  # (suffix, tag)
            "f103",  # (prefix, tag)
            "f104",  # (prev_prev_tag, prev_tag, cur_tag)
            "f105",  # (tag,)
            "f106",  # (prev_word, tag)
            "f107",  # (next_word, tag)
            "f108",  # (punctuation and special characters, tag)
            "f109",  # (position and structural features, tag)
            "f110",  # (contains '.', tag) 
            "f111",  # (contains '-', tag) 
            "f112",  # (word length, tag) 
            "f114",  # (word shape, tag) 
            "f115",  # (word short shape, tag) 
        ]
        self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}

        self.tags = set()
        self.tags.add("~")
        self.tags_counts = defaultdict(int)
        self.words_count = defaultdict(int)
        self.histories = []

    def get_word_tag_pair_count(self, file_path) -> None:
        with open(file_path) as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                split_words = line.split(' ')
                for word_idx in range(len(split_words)):
                    cur_word, cur_tag = split_words[word_idx].split('_')
                    self.tags.add(cur_tag)
                    self.tags_counts[cur_tag] += 1
                    self.words_count[cur_word] += 1

                    # ---------- f100 : (word, tag) ----------
                    self.feature_rep_dict["f100"][(cur_word, cur_tag)] = \
                        self.feature_rep_dict["f100"].get(
                            (cur_word, cur_tag), 0) + 1

                    # ---------- f101 : (prev_tag, cur_tag) ----------
                    prev_tag = split_words[word_idx -
                                           1].rsplit('_', 1)[1] if word_idx > 0 else "*"
                    self.feature_rep_dict["f101"][(prev_tag, cur_tag)] = \
                        self.feature_rep_dict["f101"].get(
                            (prev_tag, cur_tag), 0) + 1

                    # ---------- f102 : (suffix, tag) ----------
                    for suffix_length in [2, 3]:
                        if len(cur_word) >= suffix_length:
                            suffix = cur_word[-suffix_length:].lower()
                            self.feature_rep_dict["f102"][(suffix, cur_tag)] = \
                                self.feature_rep_dict["f102"].get(
                                    (suffix, cur_tag), 0) + 1

                    # ---------- f103 : (prefix, tag) ----------
                    for prefix_length in [2, 3]:
                        if len(cur_word) >= prefix_length:
                            prefix = cur_word[:prefix_length].lower()
                            self.feature_rep_dict["f103"][(prefix, cur_tag)] = \
                                self.feature_rep_dict["f103"].get(
                                    (prefix, cur_tag), 0) + 1

                    # ---------- f104 : (prev_prev_tag, prev_tag, cur_tag) ----------
                    if word_idx > 1:
                        prev_prev_tag = split_words[word_idx -
                                                    2].rsplit('_', 1)[1]
                    else:
                        prev_prev_tag = "*"
                    self.feature_rep_dict["f104"][(prev_prev_tag, prev_tag, cur_tag)] = \
                        self.feature_rep_dict["f104"].get(
                            (prev_prev_tag, prev_tag, cur_tag), 0) + 1

                    # ---------- f105 : (tag,) ----------
                    self.feature_rep_dict["f105"][(cur_tag,)] = \
                        self.feature_rep_dict["f105"].get((cur_tag,), 0) + 1

                    # ---------- f106 : (prev_word, tag) ----------
                    prev_word = split_words[word_idx -
                                            1].rsplit('_', 1)[0] if word_idx > 0 else "*"
                    if len(prev_word) >= 2:
                        self.feature_rep_dict["f106"][(prev_word, cur_tag)] = \
                            self.feature_rep_dict["f106"].get(
                                (prev_word, cur_tag), 0) + 1

                    # ---------- f107 : (next_word, tag) ----------
                    next_word = split_words[word_idx + 1].rsplit(
                        '_', 1)[0] if word_idx < len(split_words) - 1 else "~"
                    if len(next_word) >= 3:
                        self.feature_rep_dict["f107"][(next_word, cur_tag)] = \
                            self.feature_rep_dict["f107"].get(
                                (next_word, cur_tag), 0) + 1

                      # ---------- f108 : (capitalization_pattern, tag) ----------
                    if cur_word and cur_word[0].isupper():
                        feature_key = ("FIRST_CAPITAL", cur_tag)
                        if feature_key not in self.feature_rep_dict["f108"]:
                            self.feature_rep_dict["f108"][feature_key] = 1
                        else:
                            self.feature_rep_dict["f108"][feature_key] += 1

                    if cur_word.isupper() and len(cur_word) > 1:
                        feature_key = ("ALL_CAPITAL", cur_tag)
                        if feature_key not in self.feature_rep_dict["f108"]:
                            self.feature_rep_dict["f108"][feature_key] = 1
                        else:
                            self.feature_rep_dict["f108"][feature_key] += 1

                    # ---------- f109 : (number_pattern, tag) ----------
                    if any(c.isdigit() for c in cur_word):
                        feature_key = ("HAS_DIGIT", cur_tag)
                        if feature_key not in self.feature_rep_dict["f109"]:
                            self.feature_rep_dict["f109"][feature_key] = 1
                        else:
                            self.feature_rep_dict["f109"][feature_key] += 1

                    if cur_word.isdigit():
                        feature_key = ("IS_NUMBER", cur_tag)
                        if feature_key not in self.feature_rep_dict["f109"]:
                            self.feature_rep_dict["f109"][feature_key] = 1
                        else:
                            self.feature_rep_dict["f109"][feature_key] += 1

                    if '.' in cur_word and any(c.isdigit() for c in cur_word):
                        feature_key = ("IS_DECIMAL", cur_tag)
                        if feature_key not in self.feature_rep_dict["f109"]:
                            self.feature_rep_dict["f109"][feature_key] = 1
                        else:
                            self.feature_rep_dict["f109"][feature_key] += 1

                # ---------- f110 : (contains '.', tag) - YOUR FEATURE ----------
                if sum([l == '.' for l in cur_word]) > 0:  # contains a '.'
                    if ('.', cur_tag) not in self.feature_rep_dict['f110']:
                        self.feature_rep_dict["f110"][('.', cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f110"][('.', cur_tag)] += 1

                # ---------- f111 : (contains '-', tag) - YOUR FEATURE ----------
                if sum([l == '-' for l in cur_word]) > 0:  # contains a '-'
                    if ('-', cur_tag) not in self.feature_rep_dict['f111']:
                        self.feature_rep_dict["f111"][('-', cur_tag)] = 1
                    else:
                        self.feature_rep_dict["f111"][('-', cur_tag)] += 1

                # ---------- f112 : (word length, tag) - YOUR FEATURE ----------
                # word length
                if (str(len(cur_word)), cur_tag) not in self.feature_rep_dict["f112"]:
                    self.feature_rep_dict["f112"][(
                        str(len(cur_word)), cur_tag)] = 1
                else:
                    self.feature_rep_dict["f112"][(
                        str(len(cur_word)), cur_tag)] += 1

                # ---------- f114 : (word shape, tag) - YOUR FEATURE ----------
                shape = word_shape(cur_word)
                # word general shape
                if (shape, cur_tag) not in self.feature_rep_dict["f114"]:
                    self.feature_rep_dict["f114"][(shape, cur_tag)] = 1
                else:
                    self.feature_rep_dict["f114"][(shape, cur_tag)] += 1

                # ---------- f115 : (word short shape, tag) - YOUR FEATURE ----------
                short_shape = word_short_shape(cur_word)
                # word short shape
                if (short_shape, cur_tag) not in self.feature_rep_dict["f115"]:
                    self.feature_rep_dict["f115"][(short_shape, cur_tag)] = 1
                else:
                    self.feature_rep_dict["f115"][(short_shape, cur_tag)] += 1

                sentence = [("*", "*"), ("*", "*")]
                for pair in split_words:
                    sentence.append(tuple(pair.split("_")))
                sentence.append(("~", "~"))

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
            scores[feat_type] = {
                feat: count / total for feat, count in features.items()
            }
        return scores

    def filter_top_scoring_features(self, top_percentile=0.8, min_threshold=5):
        scores = self.compute_feature_scores()

        custom_percentile = {
            "f100": 0.3,
            "f101": 0.3,
            "f102": 0.25,
            "f103": 0.25,
            "f104": 0.3,
            "f106": 0.4,
            "f107": 0.4,
        }

        for feat_type, score_dict in scores.items():
            perc = custom_percentile.get(feat_type, top_percentile)
            sorted_feats = sorted(score_dict.items(), key=lambda x: -x[1])
            cutoff = int(len(sorted_feats) * perc)

            top_feats = set()
            for feat, _ in sorted_feats[:cutoff]:
                count = self.feature_rep_dict[feat_type].get(feat, 0)
                if count >= min_threshold:
                    top_feats.add(feat)

            self.feature_rep_dict[feat_type] = OrderedDict(
                (f, self.feature_rep_dict[feat_type][f])
                for f in self.feature_rep_dict[feat_type]
                if f in top_feats
            )


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        self.feature_statistics = feature_statistics
        self.threshold = threshold
        self.n_total_features = 0
        self.feature_to_idx = {feat: OrderedDict()
                               for feat in feature_statistics.feature_rep_dict}
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx
        """
        for feat_class in self.feature_statistics.feature_rep_dict:
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

    def calc_represent_input_with_features(self) -> None:
        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []
        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for y_tag in self.feature_statistics.tags:
                demi_hist = (hist[0], y_tag, hist[2],
                             hist[3], hist[4], hist[5], hist[6])
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1

        self.big_matrix = sparse.csr_matrix(
            (np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
            shape=(len(self.feature_statistics.tags) *
                   len(self.feature_statistics.histories), self.n_total_features),
            dtype=bool
        )

        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(self.feature_statistics.histories),
                   self.n_total_features),
            dtype=bool
        )


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple, int]]) -> List[int]:
    c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word = history
    features = []

    def add_feature(feat_class, key):
        if key in dict_of_dicts[feat_class]:
            features.append(dict_of_dicts[feat_class][key])

    add_feature("f100", (c_word, c_tag))
    add_feature("f101", (p_tag, c_tag))

    # f102: suffixes
    for suffix_length in [2, 3]:
        if len(c_word) >= suffix_length:
            suffix = c_word[-suffix_length:].lower()
            add_feature("f102", (suffix, c_tag))

    # f103: prefixes
    for prefix_length in [2, 3]:
        if len(c_word) >= prefix_length:
            prefix = c_word[:prefix_length].lower()
            add_feature("f103", (prefix, c_tag))

    add_feature("f104", (pp_tag, p_tag, c_tag))
    add_feature("f105", (c_tag,))

    # ---------- f106 ----------
    if len(p_word) >= 2:
        add_feature("f106", (p_word, c_tag))

    # ---------- f107 ----------
    if len(n_word) >= 3:
        add_feature("f107", (n_word, c_tag))

    if c_word and c_word[0].isupper():
        add_feature("f108", ("FIRST_CAPITAL", c_tag))
    if c_word.isupper() and len(c_word) > 1:
        add_feature("f108", ("ALL_CAPITAL", c_tag))

    if any(c.isdigit() for c in c_word):
        add_feature("f109", ("HAS_DIGIT", c_tag))
    if c_word.isdigit():
        add_feature("f109", ("IS_NUMBER", c_tag))
    if '.' in c_word and any(c.isdigit() for c in c_word):
        add_feature("f109", ("IS_DECIMAL", c_tag))

    # f110: contains '.'
    if sum([l == '.' for l in c_word]) > 0:
        add_feature("f110", ('.', c_tag))

    # f111: contains '-'
    if sum([l == '-' for l in c_word]) > 0:
        add_feature("f111", ('-', c_tag))

    # f112: word length
    add_feature("f112", (str(len(c_word)), c_tag))

    # f114: word shape
    add_feature("f114", (word_shape(c_word), c_tag))

    # f115: word short shape
    add_feature("f115", (word_short_shape(c_word), c_tag))

    return features


def preprocess_train(train_path, threshold):
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    statistics.filter_top_scoring_features(
        top_percentile=0.80, min_threshold=10)

    print("📊 Features after score filtering:")
    for k, v in statistics.feature_rep_dict.items():
        print(f"{k}: {len(v)}")

    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()

    print(f"\n✅ Total active features: {feature2id.n_total_features}")
    for dict_key in feature2id.feature_to_idx:
        print(f"{dict_key}: {len(feature2id.feature_to_idx[dict_key])}")

    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
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


def word_shape(word: str) -> str:
    if word.isupper():
        return "ALLCAPS"
    if word[0].isupper():
        return "INITCAP"
    if word.islower():
        return "LOWER"
    if word.isdigit():
        return "NUM"
    if any(c.isdigit() for c in word):
        return "HASDIGIT"
    return "OTHER"


def word_short_shape(word: str) -> str:
    return word_shape(word)  # reuse compact logic
