# CivicBench toolkit guide

Load the CivicBench benchmark and attack **your own** video detector with the paper's protocol:
PGD, APGD and the Square attack at budgets of a few gray levels, every candidate checked after
rounding to 8 bits. One call gives the columns of the paper's CivicBench table: clean AUC and TPR,
TPR under evasion and the false-alarm rate after framing at each budget.

License: Apache-2.0 (see `LICENSE` and `NOTICE`).

## Install

```bash
pip install .                 # core: numpy and torch only
pip install ".[data]"         # + pyarrow, Pillow, datasets to load the parquet release
pip install ".[fancy]"        # + rich for the live progress line (a built-in fallback exists)
```

Python 3.9 or newer.

## Your detector

A detector is any callable `detector(x) -> score`.

| | |
|---|---|
| input `x` | `torch.float32` tensor `[T, 3, H, W]`, RGB, values in [0, 1]. The official CivicBench input is `[11, 3, 512, 512]`: 11 frames of 512 x 512. |
| output | one score: a Python float, a 0-d tensor or a 1-element tensor. Higher means more likely generated. A clip is flagged when `score >= tau`. |
| gradients | PGD and APGD need the score to be differentiable in `x` (torch autograd). Square and SimBA need only score values. |

Put your model in eval mode, move it to its device (the clip tensor follows the first parameter
of an `nn.Module`, or pass `device=`), and do any resizing or normalization inside the callable,
so that the attack works on the pixels a viewer sees. If your score has no gradient, the gradient
attacks make no progress (a warning says so); run `attacks=("square",)` or put a differentiable
approximation inside your callable.

## Quick start

```python
import civicbench

clips = civicbench.load("path/to/civicbench")          # a directory of parquet shards, or a Hugging Face repo id
res = civicbench.evaluate(my_detector, clips)           # prints the table below; returns a dict

# one clip at a time
x = next(iter(clips)).frames                            # uint8 [11, 512, 512, 3]
r = civicbench.attack(my_detector, x, tau=res["tau"], direction="evasion", budgets=(1, 2, 4))
r.success          # {1: False, 2: True, 4: True}, monotone in the budget
r.cost             # 2.0: the lowest budget at which an attack succeeded (inf if none)
r.scores           # best score reached at each budget
r.frames_adv       # uint8 frames of the successful clip (r.adversarial: the best clip per budget)
r.per_attack       # per attack: cost, queries, trace [{budget, evaded, score, queries, seconds}, ...]

# margin against reach: predict the share that keeps its label at 4 gray levels from a 1-gray-level PGD
probes = [civicbench.attack(my_detector, c.frames, res["tau"], "evasion", budgets=(1,), attacks=("pgd",))
          for c in flagged_generated_clips]
civicbench.predict_survival(probes, budget=4, probe_budget=1)["evasion"]["predicted_keep"]
```

`load()` without a source streams both Hugging Face repos, the open real clips and the gated
generated clips (`civicbench.DEFAULT_REPO`). Accept the generated repo's terms once on its dataset
page and log in with `hf auth login`. `load` also accepts a repo id, a local directory, a single
parquet file or a list of them, and yields `Clip(clip_id, frames, label, meta)` tuples with `label` 0 = real, 1 = generated.
It checks each clip's `frames_sha256` after decoding. `evaluate` also takes any re-iterable of
`(clip_id, frames_u8, label[, meta])` tuples.

A complete run without downloads or weights (40 synthetic clips, a toy detector, under a minute on
a laptop CPU):

```bash
python examples/demo_toy.py
```

## Command line

```bash
civicbench evaluate --detector my_module:make_detector --data path/to/real path/to/generated \
    --budgets 1,2,4,8,16 --out results.json
```

`my_module:make_detector` is a factory: it is imported from the current directory, called (with
`--factory-kwargs '{"ckpt": "..."}'` if given) and must return the detector callable. Other options:
`--fpr`, `--strict-fpr`, `--attacks pgd,apgd,square`, `--n-evasion`, `--n-framing`,
`--evasion-per-generator`, `--seed`, `--survival-budget`, `--survival-probe`, `--device`, `--limit`,
`--split`, `--no-verify`, `--quiet`. `python -m civicbench` works too.

## Protocol

These are the paper's CivicBench settings; every argument can be changed.

* **Threshold.** `tau` flags 5% of the real clips (`threshold_at_fpr(real_scores, 0.05)`: the
  smallest threshold whose false-alarm rate on the real clips is at most 5%). TPR is the share of
  generated clips with `score >= tau`. AUC is computed on all clips.
* **Budgets.** l_inf budgets in gray levels on the 0-255 grid, default `(1, 2, 4)`. Any integer
  levels from 1 to 255 are allowed (sorted and deduplicated). An attack at budget b changes every
  pixel by at most b gray levels, and every candidate is rounded to 8 bits before it is scored.
* **Attacks** (all run at each budget in ascending order and stop at their first success):
  * PGD: 100 steps of b/4 gray levels along the sign of the gradient, starting at the clean clip,
    the 8-bit clip checked every 10 steps.
  * APGD (Croce and Hein, ICML 2020): 100 iterations x 5 restarts, momentum 0.75, initial step 2b,
    halved at the standard checkpoints when progress stalls (rho = 0.75); restart 0 starts at the
    clean clip, the others uniformly in the ball; the 8-bit clip checked every 10 iterations.
  * Square (Andriushchenko et al., ECCV 2020), video variant: 1,000 scored queries per budget;
    each query sets a square patch of one random frame to the clean pixels plus or minus b
    (a random sign per channel) and keeps it if the score improves; the standard schedule of
    patch sizes (p_init = 0.2).
  * SimBA-DCT (Guo et al., ICML 2019), optional (`attacks=(..., "simba")`): 1,000 queries per budget
    along low-frequency DCT directions of single frames and channels.
* **Evasion** attacks the flagged generated clips and succeeds at budget b when an 8-bit clip within
  b gray levels scores below `tau`. TPR under evasion at b = clean TPR x share of the attacked clips
  that no attack breaks at b or below.
* **Framing** attacks real clips. The attacks aim at the stricter threshold for 1% false alarms
  (`strict_fpr=0.01`), but a clip counts as framed at budget b once any attack reaches `tau` at b
  or below. A real clip flagged before the attack counts as framed at every budget. Attacked
  false-alarm rate at b = share of all attacked real clips framed at b.
* **Per clip,** the attack that succeeds at the lowest budget decides, and success is made monotone
  in the budget. Every claimed success is re-verified on the saved 8-bit clip.
* **Seeds.** Attack `a` on clip `c` draws from `numpy.random.default_rng(seed_for("civic", 0, c, a))`,
  `seed_for` being the first 32 bits of SHA-1 over the colon-joined parts, as in the paper's runs.
  `evaluate(seed=s)` uses `(s, c, a)` instead; `attack(seed=...)` takes an int or a tuple of parts.

## What `evaluate` returns

A dict with `n_real`, `n_generated`, `tau`, `tau_strict`, `auc`, `tpr`, `budgets`, `attacks`,
`n_evasion`, `n_framing`, `evasion_tpr` (budget -> TPR under evasion), `evasion_success`
(budget -> share of attacked flagged clips broken), `framing_rate` (budget -> false-alarm rate after
framing), `framed_clean` (share of the attacked real clips flagged before the attack), `survival`
(the margin-against-reach prediction, below), `seconds` and `clips` (one JSON-friendly record per
attacked clip, with every attack's per-budget trace). The printed summary:

```
  AUC    TPR    e@1    e@2    e@4    f@1    f@2    f@4
0.970  0.600  0.500  0.300  0.050  0.050  0.150  0.550
```

with one `e@b` (TPR under evasion) and one `f@b` (false-alarm rate after framing) column per
budget. By default every flagged generated clip and every real clip is attacked. The paper attacked
up to 10 flagged generated clips per generator and 80 real clips for most detectors
(`evasion_per_generator=10, n_framing=80`); with such caps the attacked rates are estimates on a
subset, in clip order.

## Margin against reach

For an attacked clip x with clean score s(x) and threshold `tau`, the margin is m = s(x) - tau for
a generated clip (evasion) and tau - s(x) for a real one (framing). The reach r(b) is how far the
attack moved the score toward the threshold at b gray levels: s(x) - s(x_b) for evasion, s(x_b) -
s(x) for framing, x_b being the best clip PGD found at b. A clip breaks at b when r(b) >= m. Since
the reach grows almost linearly in the budget, a probe at budget p predicts that a clip breaks at
budget B when (B / p) r(p) >= m; the paper uses p = 1 and B = 4, so 4 r(1) >= m. The break budget
p m / r(p) is the budget at which the clip is predicted to break.

`predict_survival(results, budget=4, probe_budget=1)` returns, per direction: `n`,
`predicted_keep`, `measured_keep` (from the same attack's score at `budget`, when it was run),
`error_pp` (predicted minus measured share broken, in percentage points), `break_budget_median`,
`break_budget_q10`, `margin_median`, `reach_median`, `reach_ratio_median` (median r(B) / r(p)) and
per-clip rows. The measured reach is a lower bound on the true reach (an attack finds one reachable
clip, not necessarily the worst), so the predicted share that keeps its label is an upper bound.

## Progress display

Long steps (loading, attacking a clip, evaluating) show ONE live line on a terminal, redrawn in
place, only after about 2 seconds: a spinner, a block bar, percent, done/total, elapsed time, ETA
and the current stage, for example `framing @4 · square · evaded 11/12 · framed 3/20 @4`. It ends
with one summary line. `rich` draws it when installed, otherwise a built-in fallback does.
`NO_COLOR` turns colors off. When stdout is not a terminal (cluster logs), one compact line is
printed every 10% of the work or every 5 minutes, whichever is rarer. `quiet=True` (`--quiet`)
prints nothing.

## Tests

```bash
python -m unittest discover -s tests -t tests      # or: pytest tests
```

The tests use tiny clips (3 frames of 16 x 16) and a toy linear detector: budget bounds on the
8-bit grid for default, custom and single budgets, direction of the score change, monotone
success, determinism, the threshold and AUC against brute-force definitions, `evaluate` columns,
the survival predictor, loading a synthetic parquet shard, the command line, and the progress
display. A golden test compares the attacks with values recorded from the paper's original
implementation on the same toy setting.

## Fidelity notes

What differs from the code behind the paper, and why.

1. **Detector interface.** The paper's harness scored each candidate through the detector
   pipeline's own path (for some detectors a video codec) and took gradients through the
   pipeline's differentiable path: exact gradients for frame-based detectors such as Tau-Anchor,
   and smooth stand-ins (BPDA) for detectors that read codec motion vectors. Here the detector is
   your callable: candidates are rounded to 8 bits, converted back to `[T, 3, H, W]` floats and
   scored; gradients are those of your callable at the unrounded iterate, as for the paper's
   frame-based detectors. The motion-vector stand-ins are not included.
2. **Attacks ported unchanged**: PGD, APGD, Square, SimBA-DCT, the cost-to-evade sweep (stop at
   the first realized success, re-verify every success) and the realized-query bookkeeping. Step
   sizes, iteration counts, restarts, query budgets, check intervals and the Square schedule are
   the paper's. Not included: NES, HopSkipJump, the EOT variant of APGD, Square with a gradient
   prior and the control attacks of the paper's ablations. Square's `prior` argument was removed.
3. **Budgets** may be any integer gray levels from 1 to 255. The source has no table keyed on
   particular budgets: PGD's step (b/4), APGD's initial step (2b), Square's patch values (plus or
   minus b) and SimBA's step (b) all scale with eps = b / 255 exactly as in the source, and Square's
   patch schedule depends on the query count only.
4. **Framing** negates the score and the threshold, as the source's negated readout did. The
   source's evasion test `score < tau` thus becomes `score > tau` for the attack's stopping rule,
   while success is counted at `score >= tau` from the per-budget scores, exactly as in the paper's
   analysis. `tau_strict` is never below `tau`.
5. **Non-finite scores** count as `+inf` on the attack's side (never a success), as a failed
   query did in the source. A non-finite score on a clean clip raises an error.
6. **Queries and batching.** The source could score batches of candidates in a thread pool (used
   only by attacks not included here); every query here is one call of your detector. The source
   cast features to float64 before its linear readout; here the detector's own dtype is used.
7. **Outcome per clip** follows the paper's analysis: the lowest budget at which any attack
   succeeds, success made monotone in the budget, and real clips flagged before any attack counted
   as framed at every budget. The analysis also re-judged attacks that ran at an older threshold;
   that step does not apply here, since every attack runs at the threshold it is judged at.
8. **Attack sets.** The paper attacked fixed lists of clips (the paired sets shared by Tau-Anchor
   and its baselines, or up to 10 flagged clips per generator and 80 real clips). Those lists are
   not shipped; `evaluate` attacks every flagged generated clip and every real clip unless you cap
   them (`n_evasion`, `n_framing`, `evasion_per_generator`, in clip order). The paper ran APGD and
   Square on subsets for some detectors; the defaults here run all three attacks on every attacked
   clip.
9. **Margin against reach.** The prediction (4 r(1) >= m, generalized to (B / p) r(p) >= m), the
   break budget m / r(1) (generalized to p m / r(p)), the reach ratio and the measured breaks under
   evasion match the paper's analysis. One deliberate difference: for framing, the analysis counted
   a clip whose attack had already crossed the stricter 1% threshold at a lower budget (so it has
   no score at 4) as not broken at 4; here it counts as broken, since it is framed at every larger
   budget. If an attack succeeds below the probe budget, the reach at that lower budget serves as
   the clip's probe.
10. **Not included**: perceptual metrics of adversarial clips (PSNR, LPIPS, SSIM, VMAF), saving
    adversarial deltas, resumable part files and per-check logs of the paper's cluster runs, and
    decoding clips from video files (the release ships the official 11-frame input directly).
11. **AUC** is computed with the rank formula (ties count one half), equal to scikit-learn's
    `roc_auc_score`, so the core needs only numpy and torch.

Checked equivalence: on toy detectors (linear and nonlinear) and tiny clips, the ported attacks
reproduce the original implementation exactly, for every attack, budget and direction: the same
cost, the same number of queries, the same per-budget scores (to 1e-9) and bit-identical
adversarial clips. The golden test in `tests/test_attacks.py` keeps that check.
