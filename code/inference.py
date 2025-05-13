# inference.py

from tqdm import tqdm
from math import log
import numpy as np

BEAM_SIZE   = 5
SOFT_MARGIN = 5.0

def memm_viterbi(sentence, pre_trained_weights, feature2id, represent_input_with_features):
    """
    Applies the Viterbi algorithm with beam search and heuristic pruning to decode the best tag sequence for a sentence.

    This function is tailored for a Maximum Entropy Markov Model (MEMM) and uses:
    - Beam search to limit computation (only top N states are kept at each step).
    - A soft margin to prune low-scoring paths (heuristic pruning).

    Args:
        sentence (List[str]): The input sentence as a list of words.
        pre_trained_weights (np.ndarray): The trained weight vector for features.
        feature2id (FeatureIndexer): Object holding feature-to-index and tag mappings.
        represent_input_with_features (Callable): Function to convert a history tuple to a list of active feature indices.

    Returns:
        List[str]: The most probable sequence of tags for the input sentence.
    """
    n = len(sentence)
    beam = [{} for _ in range(n + 1)]
    beam[0][("*", "*")] = (0.0, [])

    for t in range(1, n + 1):
        cur_word = sentence[t - 1]
        beam_t = {}

        for (prev_prev_tag, prev_tag), (prev_score, prev_path) in beam[t - 1].items():
            for curr_tag in sorted(feature2id.feature_statistics.tags):
                history = (
                    cur_word, curr_tag,
                    sentence[t - 2] if t >= 2 else "*", prev_tag,
                    sentence[t - 3] if t >= 3 else "*", prev_prev_tag,
                    sentence[t] if t < n else "~"
                )

                features = represent_input_with_features(history, feature2id.feature_to_idx)
                score = prev_score + sum(pre_trained_weights[f] for f in features)
                new_path = prev_path + [curr_tag]

                state = (prev_tag, curr_tag)
                if state not in beam_t or score > beam_t[state][0]:
                    beam_t[state] = (score, new_path)

        best_score = max(val[0] for val in beam_t.values())
        pruned = {
            state: val
            for state, val in beam_t.items()
            if val[0] >= best_score - SOFT_MARGIN
        }

        beam[t] = dict(list(pruned.items())[:BEAM_SIZE])

    best_score, best_path = max(beam[n].values(), key=lambda x: x[0])
    return best_path

def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path, read_test, represent_input_with_features):
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    with open(predictions_path, "w") as output_file:
        for sentence, _ in test:
            pred = memm_viterbi(sentence, pre_trained_weights, feature2id, represent_input_with_features)
            core_words = sentence[2:-1]
            core_pred = pred[2:-1]
            output_file.write(" ".join(f"{w}_{t}" for w, t in zip(core_words, core_pred)) + "\n")
