import numpy as np


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores)
    k = min(k, scores.shape[-1])
    if k == scores.shape[-1]:
        return np.broadcast_to(np.arange(k), scores.shape[:-1] + (k,))
    return np.argpartition(-scores, k - 1, axis=-1)[..., :k]


def recall_at_k(scores: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    scores = np.atleast_2d(scores)
    truth = np.atleast_2d(np.asarray(truth, dtype=np.int64))
    n, experts = scores.shape
    rows = np.arange(n)[:, None]
    mask = np.zeros((n, experts), dtype=bool)
    mask[rows, truth] = True
    hits = mask[rows, top_k(scores, k)].sum(-1)
    return hits / mask.sum(-1)
