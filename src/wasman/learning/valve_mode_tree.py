"""Small class-balanced CART with quantile split candidates; no extra dependency."""

import numpy as np
import torch


def fit_mode_tree(features, labels, *, max_depth=8, min_leaf=32, split_candidates=256, num_classes=4):
    if num_classes < 2 or max_depth < 0 or min_leaf < 1 or split_candidates < 2:
        raise ValueError("Invalid tree fitting parameters")
    x, y = np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all() or set(np.unique(y)) != set(range(num_classes)):
        raise ValueError("Finite feature matrix and all requested training modes are required")
    class_weight = len(y) / (num_classes * np.bincount(y, minlength=num_classes))
    nodes = []

    def grow(ids, depth):
        counts = np.bincount(y[ids], minlength=num_classes) * class_weight
        index = len(nodes)
        nodes.append([-1, -1, -1, 0.0, int(counts.argmax())])
        if depth >= max_depth or len(ids) < 2 * min_leaf or np.count_nonzero(counts) == 1:
            return index
        best_score = np.square(counts).sum() / counts.sum()
        best = None
        for feature in range(x.shape[1]):
            order = ids[np.argsort(x[ids, feature], kind="stable")]
            values = x[order, feature]
            sampled = np.unique(
                values[np.linspace(0, len(order) - 1, min(len(order), split_candidates + 1)).astype(int)]
            )
            if len(sampled) < 2:
                continue
            thresholds = (sampled[:-1] + sampled[1:]) / 2
            positions = np.searchsorted(values, thresholds, side="right")
            valid = (positions >= min_leaf) & (positions <= len(order) - min_leaf)
            positions, thresholds = positions[valid], thresholds[valid]
            if not len(positions):
                continue
            cumulative = np.cumsum(np.eye(num_classes)[y[order]] * class_weight, axis=0)
            left = cumulative[positions - 1]
            right = counts - left
            score = np.square(left).sum(1) / left.sum(1) + np.square(right).sum(1) / right.sum(1)
            winner = int(score.argmax())
            if score[winner] > best_score + 1e-10:
                best_score = score[winner]
                best = (feature, float(thresholds[winner]))
        if best is not None:
            feature, threshold = best
            selected = x[ids, feature] <= threshold
            left, right = grow(ids[selected], depth + 1), grow(ids[~selected], depth + 1)
            nodes[index][:4] = [left, right, feature, threshold]
        return index

    grow(np.arange(len(y)), 0)
    arrays = np.asarray(nodes)
    return {
        **{
            name: torch.tensor(arrays[:, i], dtype=torch.float64 if name == "threshold" else torch.long)
            for i, name in enumerate(("left", "right", "feature", "threshold", "prediction"))
        },
        "max_depth": max_depth,
        "min_leaf": min_leaf,
        "split_candidates": split_candidates,
        "algorithm": "class-balanced CART with quantile threshold candidates",
    }
