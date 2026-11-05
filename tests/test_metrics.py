"""Operating point and AUC against brute-force definitions."""
import unittest

import numpy as np

from civicbench.metrics import auc, threshold_at_fpr


def brute_threshold(real, fpr):
    """Smallest tau among the real scores (and just above the largest) with mean(real >= tau) <= fpr."""
    real = np.asarray(real, float)
    allowed = int(np.floor(fpr * real.size + 1e-9))
    cands = sorted(set(real.tolist())) + [float(np.nextafter(real.max(), np.inf))]
    return min(t for t in cands if (real >= t).sum() <= allowed)


def brute_auc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


class TestThreshold(unittest.TestCase):
    def test_matches_definition_on_random_scores(self):
        rng = np.random.default_rng(0)
        for i in range(400):
            n = int(rng.integers(1, 200))
            real = rng.normal(size=n) if i % 2 else rng.integers(0, 15, size=n).astype(float)   # with ties
            for fpr in (0.0, 0.01, 0.05, 0.1, 0.33, 1.0):
                self.assertEqual(threshold_at_fpr(real, fpr), brute_threshold(real, fpr))

    def test_operating_point(self):
        real = np.arange(100, dtype=float)
        tau = threshold_at_fpr(real, 0.05)
        self.assertEqual(tau, 95.0)
        self.assertAlmostEqual(float(np.mean(real >= tau)), 0.05)
        self.assertGreater(threshold_at_fpr(real, 0.0), 99.0)

    def test_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            threshold_at_fpr([0.0, np.nan], 0.05)
        with self.assertRaises(ValueError):
            threshold_at_fpr([], 0.05)


class TestAUC(unittest.TestCase):
    def test_matches_pairwise_definition(self):
        rng = np.random.default_rng(1)
        for i in range(50):
            n = int(rng.integers(2, 60))
            y = rng.integers(0, 2, size=n)
            y[0], y[1] = 0, 1
            s = rng.integers(0, 6, size=n).astype(float) if i % 2 else rng.normal(size=n) + y
            self.assertAlmostEqual(auc(s, y), brute_auc(s, y), places=12)

    def test_matches_sklearn_if_available(self):
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, size=300)
        s = np.round(rng.normal(size=300) + 0.7 * y, 1)
        self.assertAlmostEqual(auc(s, y), float(roc_auc_score(y, s)), places=12)


if __name__ == "__main__":
    unittest.main()
