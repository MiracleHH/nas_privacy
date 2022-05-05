import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MemGuard(nn.Module):
    def __init__(self):
        super(MemGuard, self).__init__()

    def forward(self, input, logits=True):
        if logits:
            scores = F.softmax(input, dim=1)
        n_classes = scores.shape[1]
        epsilon = 1e-3
        on_score = (1. / n_classes) + epsilon
        off_score = (1. / n_classes) - (epsilon / (n_classes - 1))
        predicted_labels = scores.max(1)[1]
        defended_scores = torch.ones_like(scores) * off_score
        defended_scores[np.arange(len(defended_scores)),
                        predicted_labels] = on_score
        return defended_scores


def memguard(scores):
    """ Given confidence vectors, perform memguard post processing to protect from membership inference.
    Note that this defense assumes the strongest defender that can make arbitrary changes to the confidence vector
    so long as it does not change the label. We as well have the (weaker) constrained optimization that will be
    released at a future data.
    Args:
    scores: confidence vectors as 2d numpy array
    Returns: 2d scores protected by memguard.
    """
    n_classes = scores.shape[1]
    epsilon = 1e-3
    on_score = (1. / n_classes) + epsilon
    off_score = (1. / n_classes) - (epsilon / (n_classes - 1))
    predicted_labels = np.argmax(scores, axis=-1)
    defended_scores = np.ones_like(scores) * off_score
    defended_scores[np.arange(len(defended_scores)),
                    predicted_labels] = on_score
    return defended_scores
