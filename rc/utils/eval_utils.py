import numpy as np
import re
import string
from collections import Counter


################################################################################
# Text Processing Helper Functions #
################################################################################


def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.history = []
        self.last = None
        self.val = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.last = self.mean()
        self.history.append(self.last)
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def mean(self):
        if self.count == 0:
            return 0.
        return self.sum / self.count


def compute_eval_metric(eval_metric, predictions, ground_truths, cross_eval=True):
    fns = {'f1': compute_f1_score,
           'em': compute_em_score}

    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(normalize_text(prediction), normalize_text(ground_truth))
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    values = []
    for prediction, ground_truth_set in zip(predictions, ground_truths):
        if cross_eval and len(ground_truth_set) > 1:
            _scores = []
            for i in range(len(ground_truth_set)):
                _ground_truth_set = []
                for j in range(len(ground_truth_set)):
                    if j != i:
                        _ground_truth_set.append(ground_truth_set[j])
                _scores.append(metric_max_over_ground_truths(fns[eval_metric], prediction, _ground_truth_set))
            value = np.mean(_scores)
        else:
            value = metric_max_over_ground_truths(fns[eval_metric], prediction, ground_truth_set)
        values.append(value)
    return np.mean(values)


def compute_f1_score(prediction, ground_truth):
    common = Counter(prediction.split()) & Counter(ground_truth.split())
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction.split())
    recall = 1.0 * num_same / len(ground_truth.split())
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_em_score(prediction, ground_truth):
    return 1.0 if prediction == ground_truth else 0.0
