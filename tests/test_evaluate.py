"""evaluate() and the margin-against-reach predictor on toy clips."""
import contextlib
import io
import math
import unittest

import numpy as np

import civicbench
from civicbench.metrics import threshold_at_fpr

import _toy


def _evaluate(budgets, **kw):
    det = _toy.linear_detector()
    clips = _toy.dataset(12, 12)
    kw.setdefault("quiet", True)
    return det, clips, civicbench.evaluate(det, clips, budgets=budgets, n_evasion=6, n_framing=6, **kw)


class TestEvaluate(unittest.TestCase):
    def _check(self, res, clips, det, budgets):
        for k in ("n_real", "n_generated", "tau", "tau_strict", "auc", "tpr", "budgets", "n_evasion", "n_framing",
                  "evasion_tpr", "evasion_success", "framing_rate", "framed_clean", "clips"):
            self.assertIn(k, res)
        self.assertEqual(res["budgets"], list(budgets))
        self.assertEqual(sorted(res["evasion_tpr"]), list(budgets))
        self.assertEqual(sorted(res["framing_rate"]), list(budgets))
        real = [civicbench.score(det, f) for _, f, lab, _ in clips if lab == 0]
        self.assertEqual(res["tau"], threshold_at_fpr(real, 0.05))
        self.assertTrue(0.5 <= res["auc"] <= 1.0)
        e = [res["evasion_tpr"][b] for b in budgets]
        f = [res["framing_rate"][b] for b in budgets]
        self.assertTrue(all(0 <= v <= res["tpr"] + 1e-12 for v in e))
        self.assertEqual(e, sorted(e, reverse=True))          # TPR under evasion falls with the budget
        self.assertEqual(f, sorted(f))                        # framing rises with the budget
        self.assertLessEqual(res["n_evasion"], 6)
        self.assertEqual(res["n_framing"], 6)
        for r in res["clips"]:
            vals = [r["success"][b] for b in budgets]
            self.assertEqual(vals, sorted(vals))

    def test_default_budgets(self):
        det, clips, res = _evaluate((1, 2, 4))
        self._check(res, clips, det, (1, 2, 4))
        self.assertIn("survival", res)
        for d in ("evasion", "framing"):
            s = res["survival"][d]
            self.assertEqual((s["probe_budget"], s["budget"]), (1, 4))
            self.assertTrue(0 <= s["predicted_keep"] <= 1)
            # linear detector: the reach is exactly linear, so the prediction is exact
            self.assertAlmostEqual(s["predicted_keep"], s["measured_keep"])
            self.assertAlmostEqual(s["reach_ratio_median"], 4.0, places=6)

    def test_custom_budgets(self):
        det, clips, res = _evaluate((16, 3, 8))
        self._check(res, clips, det, (3, 8, 16))
        self.assertEqual((res["survival"]["evasion"]["probe_budget"], res["survival"]["evasion"]["budget"]), (3, 16))

    def test_single_budget(self):
        det, clips, res = _evaluate((6,))
        self._check(res, clips, det, (6,))
        self.assertNotIn("survival", res)

    def test_clean_only(self):
        det = _toy.linear_detector()
        res = civicbench.evaluate(det, _toy.dataset(8, 8), budgets=(), quiet=True)
        self.assertIn("auc", res)
        self.assertNotIn("evasion_tpr", res)

    def test_summary_has_one_column_per_budget(self):
        det = _toy.linear_detector()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            civicbench.evaluate(det, _toy.dataset(8, 8), budgets=(1, 3, 9), attacks=("pgd",), n_evasion=2,
                                n_framing=2)
        out = buf.getvalue()
        for b in (1, 3, 9):
            self.assertIn(f"e@{b}", out)
            self.assertIn(f"f@{b}", out)
        self.assertIn("AUC", out)

    def test_quiet_prints_nothing(self):
        det = _toy.linear_detector()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            civicbench.evaluate(det, _toy.dataset(6, 6), attacks=("pgd",), n_evasion=2, n_framing=2, quiet=True)
        self.assertEqual(buf.getvalue(), "")

    def test_bad_inputs(self):
        det = _toy.linear_detector()
        with self.assertRaises(ValueError):
            civicbench.evaluate(det, [c for c in _toy.dataset(4, 4) if c[2] == 0], quiet=True)
        with self.assertRaises(ValueError):
            civicbench.evaluate(det, _toy.dataset(4, 4), budgets=(0, 300), quiet=True)


class TestSurvival(unittest.TestCase):
    def test_predict_from_attack_results(self):
        det = _toy.linear_detector()
        u = _toy.unit_reach(det)
        results = []
        for k in range(8):
            x = _toy.clip(40 + k)
            s0 = civicbench.score(det, x)
            results.append(civicbench.attack(det, x, s0 - (0.5 + k) * u, "evasion", (1, 2, 4, 8), ("pgd",), quiet=True))
        p = civicbench.predict_survival(results, budget=4, probe_budget=1)["evasion"]
        self.assertEqual(p["n"], 8)
        # margins 0.5u .. 7.5u, reach b u: at 4 gray levels clips with margin < 4u break (k = 0..3)
        self.assertAlmostEqual(p["predicted_keep"], 0.5)
        self.assertAlmostEqual(p["measured_keep"], 0.5)
        self.assertAlmostEqual(p["break_budget_median"], 4.0)
        q = civicbench.predict_survival(results, budget=8, probe_budget=2)["evasion"]
        self.assertAlmostEqual(q["predicted_keep"], 0.0)
        self.assertAlmostEqual(q["measured_keep"], 0.0)
        r = civicbench.predict_survival([x.to_record() for x in results], budget=6, probe_budget=1)["evasion"]
        self.assertIsNone(r["measured_keep"])                 # 6 was not run
        self.assertAlmostEqual(r["predicted_keep"], 2 / 8)
        with self.assertRaises(ValueError):
            civicbench.predict_survival(results, budget=1, probe_budget=4)

    def test_margin_reach_definition(self):
        det = _toy.linear_detector()
        u = _toy.unit_reach(det)
        x = _toy.clip(60)
        s0 = civicbench.score(det, x)
        r = civicbench.attack(det, x, s0 + 9.5 * u, "framing", (1, 2, 4), ("pgd",), quiet=True)
        m, reach, stopped = civicbench.margin_reach(r)
        self.assertAlmostEqual(m, 9.5 * u)
        for b in (1, 2, 4):
            self.assertAlmostEqual(reach[b], b * u, places=7)
        self.assertIsNone(stopped)
        self.assertTrue(math.isinf(r.cost))


if __name__ == "__main__":
    unittest.main()
