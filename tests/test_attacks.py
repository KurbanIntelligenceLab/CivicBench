"""Attacks on a toy linear detector with tiny clips (3 frames of 16 x 16)."""
import math
import unittest
import warnings

import numpy as np

import civicbench
from civicbench.frames import linf_levels
from civicbench.sweep import check_budgets

import _toy

ALL = ("pgd", "apgd", "square", "simba")
INF = math.inf

# Reference values recorded with the paper's original implementation on the same detector, clips,
# thresholds and seeds: (clip, direction, attack, cost in gray levels, realized queries, best score).
GOLDEN = [
    (0, "evasion", "pgd", 1, 1, 0.5574081643), (0, "evasion", "apgd", 1, 1, 0.5574081643),
    (0, "evasion", "square", 2, 1076, 0.5687663373), (0, "evasion", "simba", 1, 35, 0.5686800567),
    (0, "framing", "pgd", 1, 1, 0.6031427042), (0, "framing", "apgd", 1, 1, 0.6031427042),
    (0, "framing", "square", 4, 2311, 0.6010027052), (0, "framing", "simba", 1, 117, 0.6010218453),
    (1, "evasion", "pgd", 2, 11, 0.5703261802), (1, "evasion", "apgd", 2, 51, 0.5703261802),
    (1, "evasion", "square", 4, 2980, 0.5816848713), (1, "evasion", "simba", 2, 1050, 0.5816601029),
    (1, "framing", "pgd", 2, 11, 0.6617952591), (1, "framing", "apgd", 2, 51, 0.6617952591),
    (1, "framing", "square", INF, 3000, 0.6536678103), (1, "framing", "simba", 2, 1116, 0.6600793033),
    (2, "evasion", "pgd", 4, 21, 0.5005068605), (2, "evasion", "apgd", 4, 101, 0.5005068605),
    (2, "evasion", "square", INF, 3000, 0.5547721025), (2, "evasion", "simba", 4, 2031, 0.5270555606),
    (2, "framing", "pgd", 4, 21, 0.6834450167), (2, "framing", "apgd", 4, 101, 0.6834450167),
    (2, "framing", "square", INF, 3000, 0.6264709687), (2, "framing", "simba", 4, 2033, 0.6589406541),
    (3, "evasion", "pgd", 4, 21, 0.4897705809), (3, "evasion", "apgd", 4, 101, 0.4897705809),
    (3, "evasion", "square", INF, 3000, 0.5366490284), (3, "evasion", "simba", 4, 2108, 0.4948408739),
    (3, "framing", "pgd", 4, 21, 0.6727087377), (3, "framing", "apgd", 4, 101, 0.6727087377),
    (3, "framing", "square", INF, 3000, 0.6163230576), (3, "framing", "simba", 4, 2126, 0.6727087377),
    (4, "evasion", "pgd", INF, 30, 0.4730405342), (4, "evasion", "apgd", INF, 150, 0.4730405342),
    (4, "framing", "pgd", INF, 30, 0.6559786904), (4, "framing", "apgd", INF, 150, 0.6559786904),
]
OFFSETS = [0.5, 1.5, 2.5, 3.5, 6.0]     # threshold distance from the clean score, in units of the 1-level reach


def _setting(k, direction):
    det = _toy.linear_detector()
    x = _toy.clip(k)
    s0, u = civicbench.score(det, x), _toy.unit_reach(det)
    if direction == "evasion":
        return det, x, s0 - OFFSETS[k] * u, None
    tau = s0 + OFFSETS[k] * u
    return det, x, tau, tau + 0.4 * u


def _strip(rec):
    """A record without wall-clock times."""
    rec = dict(rec)
    rec["per_attack"] = {k: {kk: ([{x: y for x, y in t.items() if x != "seconds"} for t in vv] if kk == "trace" else vv)
                             for kk, vv in v.items() if kk != "seconds"} for k, v in rec["per_attack"].items()}
    return rec


class TestBudgets(unittest.TestCase):
    def test_check_budgets(self):
        self.assertEqual(check_budgets((4, 1, 2, 2)), (1, 2, 4))
        self.assertEqual(check_budgets([16, 3, 8]), (3, 8, 16))
        self.assertEqual(check_budgets((6,)), (6,))
        self.assertEqual(check_budgets(6), (6,))
        self.assertEqual(check_budgets((np.int64(2), 4.0)), (2, 4))
        for bad in ((0,), (256,), (-1,), (1.5,), ("2",), (True,), (), None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                check_budgets(bad)

    def _check_grid(self, budgets, attacks=ALL):
        det = _toy.linear_detector()
        for k in range(4):
            x = _toy.clip(k)
            s0, u = civicbench.score(det, x), _toy.unit_reach(det)
            for direction, tau, stop in (("evasion", s0 - 1.7 * u, None), ("framing", s0 + 1.7 * u, s0 + 2.0 * u)):
                r = civicbench.attack(det, x, tau, direction, budgets, attacks, seed=k, stop_tau=stop, quiet=True)
                self.assertEqual(r.budgets, tuple(sorted(set(budgets))))
                self.assertTrue(r.adversarial, "no adversarial frames kept")
                for b, fr in r.adversarial.items():
                    self.assertEqual(fr.dtype, np.uint8)
                    self.assertEqual(fr.shape, x.shape)
                    self.assertLessEqual(linf_levels(fr, x), b, f"budget {b} exceeded")
                    s = civicbench.score(det, fr)
                    self.assertAlmostEqual(s, r.scores[b], places=9)        # the reported score is the realized one
                if r.frames_adv is not None:
                    s = civicbench.score(det, r.frames_adv)
                    self.assertTrue(s < tau if direction == "evasion" else s >= tau)
                vals = [r.success[b] for b in r.budgets]
                self.assertEqual(vals, sorted(vals), "success must be monotone in the budget")
                if math.isfinite(r.cost):
                    self.assertTrue(r.success[int(r.cost)])

    def test_budget_bound_default(self):
        self._check_grid((1, 2, 4))

    def test_budget_bound_custom(self):
        self._check_grid((16, 3, 8))

    def test_budget_bound_single(self):
        self._check_grid((6,))

    def test_single_attacks_respect_budget(self):
        for a in ALL:
            self._check_grid((2, 5), attacks=(a,))


class TestDirections(unittest.TestCase):
    def test_evasion_lowers_framing_raises(self):
        det = _toy.linear_detector()
        u = _toy.unit_reach(det)
        for k in range(3):
            x = _toy.clip(10 + k)
            s0 = civicbench.score(det, x)
            ev = civicbench.attack(det, x, s0 - 50 * u, "evasion", quiet=True)      # out of reach: runs every budget
            fr = civicbench.attack(det, x, s0 + 50 * u, "framing", quiet=True)
            for b in (1, 2, 4):
                self.assertLess(ev.scores[b], s0)
                self.assertGreater(fr.scores[b], s0)
                self.assertFalse(ev.success[b] or fr.success[b])
            self.assertLess(ev.scores[4], ev.scores[1])
            self.assertGreater(fr.scores[4], fr.scores[1])

    def test_monotone_many(self):
        det = _toy.linear_detector()
        u = _toy.unit_reach(det)
        for budgets in ((1, 2, 4), (3, 8, 16), (6,), (1, 2, 3, 4, 6, 8)):
            for k in range(6):
                x = _toy.clip(20 + k)
                s0 = civicbench.score(det, x)
                for direction, tau in (("evasion", s0 - (k + 0.5) * u), ("framing", s0 + (k + 0.5) * u)):
                    r = civicbench.attack(det, x, tau, direction, budgets, ("pgd", "square"), seed=k, quiet=True)
                    vals = [r.success[b] for b in r.budgets]
                    self.assertEqual(vals, sorted(vals))
                    # the linear detector's reach is exactly b * u: PGD must break the clip iff b * u > margin
                    for b in r.budgets:
                        if b * u > (k + 0.5) * u * 1.0001:
                            self.assertTrue(r.success[b])

    def test_status_and_validation(self):
        det = _toy.linear_detector()
        x = _toy.clip(0)
        s0 = civicbench.score(det, x)
        r = civicbench.attack(det, x, s0 + 1.0, "evasion", quiet=True)
        self.assertEqual(r.status, "not_flagged")
        r = civicbench.attack(det, x, s0 - 1.0, "framing", quiet=True)
        self.assertEqual(r.status, "already_flagged")
        self.assertTrue(all(r.success.values()))
        with self.assertRaises(ValueError):
            civicbench.attack(det, x, s0, "sideways")
        with self.assertRaises(ValueError):
            civicbench.attack(det, x, s0, "evasion", attacks=("fgsm",))
        with self.assertRaises(ValueError):
            civicbench.attack(det, x, s0 + 0.1, "framing", stop_tau=s0)
        with self.assertRaises(TypeError):
            civicbench.attack(det, x.astype(np.float32), s0)

    def test_no_gradient_warns(self):
        det = _toy.linear_detector()

        def scores_only(x):                         # a float: no gradient
            return float(det(x))
        x = _toy.clip(1)
        s0 = civicbench.score(det, x)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = civicbench.attack(scores_only, x, s0 - 0.5 * _toy.unit_reach(det), "evasion", attacks=("pgd", "square"),
                                  quiet=True)
        self.assertTrue(any("no gradient" in str(m.message) for m in w))
        self.assertEqual(r.per_attack["pgd"]["cost"], math.inf)        # no gradient, no progress


class TestDeterminism(unittest.TestCase):
    def test_seeded_runs_are_identical(self):
        det = _toy.linear_detector()
        x = _toy.clip(7)
        s0, u = civicbench.score(det, x), _toy.unit_reach(det)
        runs = [civicbench.attack(det, x, s0 - 3.2 * u, "evasion", (1, 2, 4), ALL, seed=("civic", 0, "a"), quiet=True)
                for _ in range(2)]
        self.assertEqual(_strip(runs[0].to_record()), _strip(runs[1].to_record()))
        for b in runs[0].adversarial:
            np.testing.assert_array_equal(runs[0].adversarial[b], runs[1].adversarial[b])
        other = civicbench.attack(det, x, s0 - 3.2 * u, "evasion", (1, 2, 4), ("square",), seed=1, quiet=True)
        self.assertNotEqual(other.per_attack["square"]["trace"][0]["score"],
                            runs[0].per_attack["square"]["trace"][0]["score"])


class TestGolden(unittest.TestCase):
    def test_matches_original_implementation(self):
        """Same detector, clips, thresholds and seeds as recorded with the paper's original code."""
        for k in range(5):
            for direction in ("evasion", "framing"):
                rows = [g for g in GOLDEN if g[0] == k and g[1] == direction]
                det, x, tau, stop = _setting(k, direction)
                r = civicbench.attack(det, x, tau, direction, (1, 2, 4), tuple(g[2] for g in rows),
                                      seed=("civic", 0, f"clip{k}"), stop_tau=stop, quiet=True)
                for _, _, a, cost, queries, best in rows:
                    pa = r.per_attack[a]
                    with self.subTest(clip=k, direction=direction, attack=a):
                        self.assertEqual(pa["cost"], cost)
                        self.assertEqual(pa["total_queries"], queries)
                        self.assertAlmostEqual(pa["final_score"], best, places=8)


if __name__ == "__main__":
    unittest.main()
