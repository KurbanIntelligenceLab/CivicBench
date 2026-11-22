# Tau-Anchor

**Tau-Anchor** is the detector of AI-generated video from the CivicBench paper. It is built to keep its verdicts when
an attacker changes each pixel by a few gray levels, in both directions: hiding a fake (**evasion**) and making real
footage look generated (**framing**).

It embeds each frame with Delta-CLIP, an image encoder pre-trained on attacked images. During training it attacks real
and generated clips. It also adds a margin term for the fakes closest to its deployed threshold, since those break
first.

This folder holds the inference code, the full training recipe and the model card.

| | |
|---|---|
| **Input** | 11 RGB frames, float in [0, 1]. The official input is 11 frames of 512 x 512 at about 8 frames per second. |
| **Output** | One score per clip. Higher means generated. A clip is flagged when its score reaches the threshold. |
| **Encoder** | Delta-CLIP ViT-H/14 at 336 px, last 4 of 32 blocks fine-tuned |
| **Released model** | Trained on folds 1 to 4 of AIGVDBench. The paper tests this model on CivicBench. |
| **Gradients** | Exact. The score is differentiable from the 8-bit pixels to the output. |

## Quick start

Run from a clone of this repository.

```bash
pip install ".[tau-anchor]"        # civicbench, torch >= 2.0, huggingface_hub
```

Score one CivicBench clip:

```python
import civicbench
from tau_anchor import TauAnchorDetector

det = TauAnchorDetector.from_pretrained(device="cuda")      # downloads the weights and the base encoder once
clip = next(iter(civicbench.load("path/to/civicbench")))    # uint8 frames [11, 512, 512, 3]
score = civicbench.score(det, clip.frames)                  # flagged when score >= det.tau
```

`det` is a regular CivicBench detector. It takes a float32 tensor `[T, 3, H, W]` in [0, 1] and returns one
differentiable score, so every attack of the toolkit runs on it:

```python
r = civicbench.attack(det, clip.frames, tau=det.tau_civicbench, direction="evasion", budgets=(1, 2, 4))
```

Evaluate it on all of CivicBench, as in the paper:

```bash
civicbench evaluate --detector tau_anchor.detector:build --data path/to/civicbench --out tau_anchor.json
```

The first call downloads the Tau-Anchor weights (about 320 MB) and the Delta-CLIP checkpoint (about 4 GB, read on the
CPU). On a GPU, scoring a clip takes under a second.

The weights are on Hugging Face at [`kurbanintelligencelab/Tau-Anchor`](https://huggingface.co/kurbanintelligencelab/Tau-Anchor).
To load a local copy, pass the file: `TauAnchorDetector.from_pretrained("tau_anchor_aigvd_fold0.pt")`.

### Thresholds

| | Value | Where it comes from |
|---|---|---|
| `det.tau` | 1.0977 | The deployed threshold, set in training at 5% false alarms on the 2,000 real test clips of AIGVDBench fold 0 |
| `det.tau_civicbench` | 1.7751 | 5% false alarms on CivicBench's real clips. The paper's CivicBench table uses this threshold. |

`civicbench.evaluate` sets its own threshold at 5% false alarms on the real clips it receives. At the deployed
threshold, Tau-Anchor flags 6.4% of CivicBench's real clips before any attack.

## Results

**CivicBench**, at the threshold that flags 5% of real clips. Evasion is the TPR under attack, higher is better.
Framing is the false-alarm rate on attacked real clips, lower is better. Budgets are in gray levels on the 0 to 255
scale. The brackets give 95% bootstrap confidence intervals.

| | AUC | TPR | Evasion 1 | Evasion 2 | Evasion 4 | Framing 1 | Framing 2 | Framing 4 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Tau-Anchor | .941 | .779 | .731 | .668 | .585 | .124 | .141 | .222 |
| 95% CI | [.936, .946] | [.761, .801] | [.705, .755] | [.631, .703] | [.540, .627] | [.081, .170] | [.092, .191] | [.163, .283] |

The attacks were PGD (100 steps), APGD (100 steps, 5 restarts) and Square (1,000 queries), with full knowledge of the
detector. Tau-Anchor faced PGD on 225 flagged generated clips and 185 real clips, and APGD and Square on a subset. An
attack counts only if the saved 8-bit clip crosses the threshold. The [leaderboard](../README.md#leaderboard) compares
it with ten other detectors.

**General benchmarks.** Five-fold means. Each fold is tested by a model trained on the other four folds of the same
benchmark. Only the AIGVDBench fold-0 model is released.

| Benchmark | AUC | TPR | Evasion 2 | Evasion 4 |
|:---|:---:|:---:|:---:|:---:|
| AIGVDBench | .988 | .935 | .931 | .883 |
| GenVidBench | .999 | .998 | .993 | .993 |

## How it works

For a clip `x` with frames `x_1, ..., x_11`:

```
e_t  = f(normalize(resize_336(x_t))) / |.|        Delta-CLIP frame embedding, unit length
e(x) = mean_t e_t / |mean_t e_t|                  clip embedding
s(x) = w . e(x) + b                               score, flagged when s(x) >= tau
```

The resize is bilinear with antialiasing, and the normalization uses the ImageNet mean and standard deviation, as in
Delta-CLIP. Averaging over frames also dilutes a perturbation confined to a few frames.

Each training step attacks the batch, then minimizes

```
CE(clean) + CE(attacked) + gamma * softplus((kappa - rho) / kappa) + lambda * |e(x) - e0(x)|^2
```

* **Attacks on both classes.** A random start within 4 gray levels and one sign step of 2 gray levels. Generated
  clips are pushed toward real, real clips toward generated.
* **Margin at the deployed threshold.** `rho = (s(x + delta) - tau_hat) / |w|` is the attacked fake's distance from
  the decision boundary in embedding units. `tau_hat` is a running estimate of the deployed threshold: the 95th
  percentile of the scores of 256 real training clips, re-estimated every 100 steps. The term is applied to attacked
  generated clips only.
* **Preservation.** `e0` is the embedding of the frozen Delta-CLIP encoder, so the fine-tuned encoder keeps its robust
  features.
* **Generator weights.** Cross-entropy and margin on generated clips are weighted per generator, more for generators
  whose clips sit near the threshold.
* **Weight averaging.** The deployed model is an exponential moving average of the trained weights.

## Training recipe

`configs/tau_anchor.json` holds these values. They are also the defaults of `python -m tau_anchor.train`.

| Setting | Value | Config key |
|:---|:---|:---|
| Base encoder | Delta-CLIP ViT-H/14, 336 px, frozen except the parts below | |
| Trained parts | Last 4 transformer blocks (29 to 32), final norm, projection, head | `blocks` |
| Head initialization | Logistic-regression probe on the frozen clip embeddings of the training folds, C = 10, balanced classes, up to 3,000 iterations | `probe_c`, `probe_max_iter` |
| Frames per clip | 2 random of 11, the same 2 for a clip in every epoch | `frames` |
| Batch size | 4 clips | `batch_size` |
| Training length | 1.5 epochs, 4,494 steps for the released model | `epochs` |
| Optimizer | AdamW, weight decay 10^-4, constant learning rates | `weight_decay` |
| Learning rate | 10^-5 encoder, 10^-3 head | `lr_encoder`, `lr_head` |
| Training attack | Uniform random start within 4 gray levels, 1 sign step of 2 gray levels, model in eval mode | `eps`, `pgd_steps` |
| Attacked classes | Real and generated | `attack_side` |
| Clean cross-entropy | Weight 1 | `clean_weight` |
| Class weight | Generated clips weighted by n_real / n_generated (2.01 for the released model) | |
| Generator weights | Median margin of all generated training clips over the generator's median margin, clipped to 1 to 3 | `generator_weights` |
| Margin term | softplus form, kappa = 0.15 in embedding units, gamma = 1 | `margin`, `kappa`, `gamma` |
| Threshold estimate | 95th percentile of 256 real training clips, every 100 steps | `tau_quantile`, `tau_clips`, `tau_every` |
| Preservation term | lambda = 1 | `lambda_preserve` |
| Weight averaging | Exponential, decay 0.999, deployed at the end | `ema_decay` |
| Precision | 16-bit mixed precision with a gradient scaler (GPU) | `amp` |
| Deployed threshold | 5% false alarms on the test fold's real clips, all 11 frames | `fpr` |
| Checkpoints | Every 200 steps, resumed on restart | `ckpt_every` |

The generator weights use the probe's out-of-fold scores on the training folds: a clip's margin is its score minus
the threshold of its fold. For the released model they were 1.83 for Wan2.1, 1.69 for CogVideoX1.5 and 1 for the
other four generators.

## Retraining

```bash
pip install ".[tau-anchor-train]"   # adds scikit-learn, PyAV and Pillow
```

**1. Prepare the clips.** The paper re-encodes every video with x264 at CRF 23 and scales it to 512 x 512, keeping the
source frame rate. Scaling does not keep the aspect ratio.

```bash
ffmpeg -i in.mp4 -c:v libx264 -crf 23 -g 250 -profile:v main -an -vf scale=512:512 out.mp4
```

The loader then reads 11 frames from the start of each clip: every k-th frame, with k the integer nearest the frame
rate divided by 8. A clip can also be a `.npy` array (uint8 `[11, 512, 512, 3]`) or a folder of 11 image frames.

**2. Write a manifest**, one CSV per benchmark:

```
clip_id,path,label,fold,generator
wan21_0001,wan21/0001.mp4,1,3,Wan2.1
real_0001,real/0001.mp4,0,3,
```

`label` is 0 for real and 1 for generated. The paper splits each benchmark into five folds grouped by prompt, so no
prompt is in both training and test data.

**3. Train.** The released model is `--fold 0` on AIGVDBench.

```bash
python -m tau_anchor.train embed    --manifest aigvd.csv --data-root /data/aigvd --cache runs/emb
python -m tau_anchor.train probe    --manifest aigvd.csv --cache runs/emb --out runs/probe.json
python -m tau_anchor.train finetune --manifest aigvd.csv --data-root /data/aigvd --cache runs/emb \
                                    --probe runs/probe.json --fold 0 --out-dir runs/fold0
python -m tau_anchor.train export   --checkpoint runs/fold0/checkpoint.pt --out tau_anchor.pt
```

* `embed` caches the frozen encoder's embeddings of all 11 frames of every clip, in 16-bit floats. They feed the probe
  and the preservation term.
* `probe` fits one linear probe per fold. The probe of the test fold starts the head. The others give the generator
  weights.
* `finetune` trains, checkpoints every 200 steps and resumes after an interruption. At the end it scores the test fold
  and writes the deployed threshold, AUC and TPR to `result.json`.
* `export` keeps the fine-tuned tensors, the head and the threshold, and drops the optimizer state.

Change any setting with `--set key=value`, for example `--set margin=false` for the ablation without the margin term.

**Compute.** Training on one fold takes about 0.5 hours on one A100 GPU (40 GB) or 1.2 hours on a V100 (32 GB). The
data loader uses 7 worker processes by default (`--set workers=...`).

**Reproducibility.** The paper's runs were not seeded, so a retrained model differs from the released one by
run-to-run variation (the paper reports three runs of the recipe). `--set seed=N` makes a run repeatable. The training
code was checked against the code behind the paper on a small random model: both give identical losses, weights and
thresholds.

## Weights

The weight file holds only the fine-tuned parts: the last 4 blocks, the final norm, the projection and the head
(about 80 million parameters in 32-bit floats). It also holds both thresholds. `from_pretrained` downloads the base
encoder from [`zw123/delta_clip_h14_336`](https://huggingface.co/zw123/delta_clip_h14_336) at a fixed revision and
loads the fine-tuned parts on top. A training checkpoint of `tau_anchor.train` loads the same way.

## Intended use

* Research on detecting AI-generated video and on its robustness to small perturbations.
* A reference robust detector for CivicBench and for new attacks.
* A screening signal for human reviewers of political video.

It is not meant to decide on its own whether a real person's footage is genuine. A score is evidence, not a verdict.

## Limitations

* The robustness is bounded by the encoder. PGD at 4 gray levels breaks 24% of the attacked political fakes.
* The threat model bounds each pixel's change. Codec-aware, warping and audio attacks need their own evaluation. The
  robustness is measured, not certified.
* The released model was trained on six open generators of AIGVDBench. Fakes from closed services, face swaps and
  lip-sync of real people may behave differently. Fakes that animate a real photo left every detector in the paper,
  Tau-Anchor included, at or below chance.
* It reads 11 frames from about the first 1.3 seconds of a clip, and no audio.
* The paper sets thresholds on each test set's own real clips, which a deployment cannot do. At the threshold set in
  training, Tau-Anchor flags 6.4% of CivicBench's real clips before any attack, against a target of 5%.
* CivicBench's real footage comes from official channels and demonstrations, not from phones or social media.

## License

The code is released under the [Apache License 2.0](../LICENSE).

The weights are derived from Delta-CLIP by Wang et al., released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Redistribution requires crediting that work (citation
below). They were trained on clips of AIGVDBench. Its generated clips follow the terms of the generators that made
them. One of them, HunyuanVideo, is licensed under the Tencent Hunyuan Community License, which restricts the
territories of use and the use of its outputs to improve other AI models.

The weights are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please credit Delta-CLIP and
CivicBench (citations below).

## Citation

```bibtex
@misc{cakiroglu2026civicbench,
  title         = {CivicBench: Robust Detection of AI-Generated Political Video},
  author        = {Cakiroglu, Mert Onur and Buxton, Elham and Dalkilic, Mehmet and Kurban, Hasan},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}

@article{wang2025double,
  title   = {Double Visual Defense: Adversarial Pre-training and Instruction Tuning for Improving Vision-Language Model Robustness},
  author  = {Wang, Zeyu and Xie, Cihang and Bartoldson, Brian and Kailkhura, Bhavya},
  journal = {arXiv preprint arXiv:2501.09446},
  year    = {2025}
}
```

Please also cite AIGVDBench (Ma et al., CVPR 2026) when you retrain on it.

## Files

```
detector.py              TauAnchorDetector and build(): the CivicBench detector, thresholds, weight location
model.py                 the encoder, the clip embedding, the head, load()
train.py                 embed, probe, finetune and export (python -m tau_anchor.train)
data.py                  manifest, clip decoding, embedding cache
configs/tau_anchor.json  the paper's recipe
```
