# DAF-Net: Complete Technical Plan
## AI-Based Restoration of Degraded Images — KLA Problem Statement (SEMICON India Hackathon 2026)

---

## Part 1: Problem Requirement Analysis

### 1.1 What is actually being asked

The task is a **blind, multi-degradation image restoration** problem, not a single-purpose denoiser or super-resolution model. A single model must reverse three coupled degradations applied in **unknown, variable order**:

| Degradation | Nature | Key modeling challenge |
|---|---|---|
| Speckle noise | Multiplicative | Scales with local intensity — noise variance is signal-dependent, not constant |
| Gaussian noise | Additive | Uniform variance, but combined with speckle it stops behaving like pure Gaussian |
| Downsampling | Deterministic resolution loss | Removes high-frequency content irrecoverably without a learned prior |

### 1.2 Constraints that shape the architecture

| Constraint (from problem statement) | Design implication |
|---|---|
| Degradation order unspecified, "may appear in any order" | Model must not assume a fixed degradation pipeline — needs to be **order-agnostic** |
| Degraded image range can exceed [0,1]; GT is strictly [0,1] | Cannot clip inputs — must preserve out-of-range signal as information |
| OOD test images from unseen structure types | Model must generalize, not memorize — regularization and augmentation are first-class design concerns, not afterthoughts |
| No architecture prescribed; pretrained weights allowed | Freedom to build on proven SOTA restoration backbones instead of designing from zero |
| Scored on accuracy (SSIM/PSNR/LPIPS) **+** throughput (H100) **+** training hygiene | This is a 3-axis optimization problem, not pure accuracy maximization — a slow SOTA model can lose to a faster, slightly less accurate one |
| Single inference script, no manual edits, must run out of the box | Engineering reliability is graded alongside the model itself |

### 1.3 Reframing the problem

This is best understood as: **"Learn a conditional restoration function that infers its own degradation context per image, then reconstructs missing spatial and frequency information, under a real-time throughput budget."**

That reframing is what the proposed architecture (DAF-Net) is built around.

---

## Part 2: Proposed Architecture — DAF-Net

**D**egradation-**A**ware **F**requency-fused Restoration **Net**work

### 2.1 Design philosophy

Rather than treating this as one generic image-to-image regression task, DAF-Net makes each of its three degradation types a first-class citizen in the architecture:

```
                    ┌──────────────────────────┐
                    │   Degradation Encoder      │
Input image ───┬───►│   (infers implicit noise/   │──► z (embedding)
(may exceed     │    │   blur/scale context)      │
 [0,1] range)    │    └──────────────────────────┘
                 │                 │
                 ▼                 │ (FiLM conditioning,
      ┌────────────────────┐       │  injected at every stage)
      │ Learnable Log-      │       │
      │ Domain Transform    │       │
      │ (de-couples         │       │
      │ multiplicative       │       │
      │ speckle from         │       │
      │ additive Gaussian)   │       │
      └────────────────────┘       │
                 │                 │
                 ▼                 ▼
      ┌─────────────────────────────────────┐
      │        Dual-Domain Backbone           │
      │  ┌───────────┐   ┌─────────────────┐ │
      │  │ Spatial    │◄─►│ Frequency        │ │
      │  │ branch     │gate│ branch (FFT/    │ │
      │  │ (NAFBlocks,│   │ recovers high-   │ │
      │  │ FiLM-      │   │ freq detail lost │ │
      │  │ conditioned)│   │ to downsampling) │ │
      │  └───────────┘   └─────────────────┘ │
      └─────────────────────────────────────┘
                 │
                 ▼
      Inverse log-domain transform
                 │
                 ▼
      Pixel-shuffle upsampling head
      (scale factor matched to 2x/4x task)
                 │
                 ▼
         Restored image (clamped to [0,1])
```

### 2.2 Component-by-component justification

| Component | Solves | Why this specific technique |
|---|---|---|
| Degradation Encoder + FiLM conditioning | Unknown/variable degradation order and severity | Learns an implicit per-image "degradation fingerprint" without needing labeled degradation types — network adapts its correction per image instead of learning one averaged correction |
| Learnable log-domain transform | Multiplicative speckle mixed with additive Gaussian | Converts multiplicative noise into a domain where it behaves additively, so a single loss/network doesn't have to fight two different noise statistics simultaneously |
| Frequency branch (FFT/wavelet) | Fine detail lost to downsampling | Spatial convolutions are biased toward local, low-frequency patterns; an explicit frequency-domain path recovers structure (edges, fine features) that spatial-only networks tend to over-smooth |
| Gated fusion (not naive addition) | OOD robustness | Lets the network learn *when* to trust frequency-domain reconstruction vs. spatial — on unfamiliar structures, frequency priors may be less reliable, so a static fusion weight would hurt generalization |
| NAFBlock backbone (SimpleGate, no nonlinear activations) | Throughput | Matches transformer-level accuracy at meaningfully lower compute — directly targets the H100 throughput score |
| Pixel-shuffle upsampling | Scale-accurate reconstruction | Avoids checkerboard artifacts common with transposed convolutions |

### 2.3 Parameter budget target

Aim for **15–25M parameters** for the full model — large enough to have capacity for the dual-domain design, small enough that a distilled version (see Part 3) comfortably clears throughput requirements on H100.

---

## Part 3: Improvements Over a Naive Baseline

| Improvement | Baseline behavior it fixes | Expected effect |
|---|---|---|
| Degradation-aware FiLM conditioning | Naive models learn one averaged correction across all degradation combinations, underperforming on any single severe case | Better accuracy across the full severity range, better OOD generalization |
| Log-domain transform for speckle | Naive L1/L2 losses implicitly assume additive noise; speckle violates this, causing systematic over/under-correction in bright regions | More uniform correction quality across intensity ranges |
| Frequency-domain branch | Pure spatial CNNs tend to output smoothed/blurry reconstructions when recovering lost resolution | Sharper fine-detail recovery, better SSIM/LPIPS on textured/structured content |
| Randomized-order, randomized-severity augmentation | Fixed-order training data causes the model to implicitly learn artifacts of that one order | Directly improves OOD generalization score |
| EMA of weights | Standard training has noisy late-stage weight updates | Free accuracy gain (~0.1–0.3 dB PSNR typical) with zero inference cost |
| Knowledge distillation to a smaller student | Full DAF-Net may be too slow for H100 throughput targets | Preserves most accuracy while cutting inference latency substantially — directly optimizes the throughput axis |
| fp16 inference + conv/batchnorm fusion | Naive fp32 inference is slower than necessary | 1.5–2x speedup with negligible accuracy loss on modern GPUs |

---

## Part 4: Overfitting Prevention Plan

Since OOD generalization is explicitly scored, overfitting prevention is not optional — treat it as a primary design axis:

1. **Data-level**: randomized-order/severity synthetic augmentation, patch-based random crop + flip/rotate, occasional severity-mixup between two degraded versions of the same ground truth.
2. **Training-level**: weight decay, stochastic depth on backbone blocks, EMA weight averaging for evaluation checkpoints.
3. **Validation-level**: construct your own held-out "pseudo-OOD" split (e.g., withhold one image source/structure type entirely from training) and monitor *that* metric — not just in-distribution validation loss — for early stopping decisions.
4. **Capacity control**: keep the model in the 15–25M parameter range rather than scaling up; the webinar speaker explicitly noted that very large models often don't justify their throughput cost for this dataset size.

---

## Part 5: Benchmarking Plan

### 5.1 Metrics to track (mirrors official evaluation)

| Metric | Tool | What it captures |
|---|---|---|
| PSNR | `skimage.metrics.peak_signal_noise_ratio` | Pixel-level fidelity |
| SSIM | `skimage.metrics.structural_similarity` | Structural/perceptual similarity |
| LPIPS | `lpips` package (AlexNet backbone, matches common practice) | Deep perceptual similarity, correlates better with human judgment than PSNR/SSIM alone |
| Inference throughput | Manual timing of full pipeline (disk read → GPU inference → disk write) | Directly maps to the official throughput benchmark |
| Parameter count / model size | `sum(p.numel() for p in model.parameters())` | Reported in your submission, informs throughput expectations |

### 5.2 Benchmark protocol

1. **Split your training data** into train / in-distribution-val / pseudo-OOD-val (hold out an entire source/structure category for the latter).
2. **Track all three metrics separately** for in-distribution vs. pseudo-OOD validation — a model that's strong in-distribution but weak OOD is exactly the failure mode the hackathon is testing for.
3. **Timing benchmark**: run inference on a batch of ~100 images, measure end-to-end wall-clock time (not just GPU compute time) — include disk I/O, since that's what the official evaluation script will measure.
4. **Ablation table**: benchmark each novelty in isolation (baseline NAFNet only → + FiLM conditioning → + log-domain transform → + frequency branch → + distillation) so you can show quantitatively what each component contributes. This is valuable both for your own decision-making and for the "Innovation & Uniqueness" slide.
5. **Compare against a simple baseline** (e.g., plain NAFNet or SwinIR without your additions) to demonstrate your novelties provide measurable lift, not just complexity for its own sake.

### 5.3 Suggested ablation table format for your slides

| Configuration | PSNR (in-dist) | PSNR (OOD) | SSIM | LPIPS | Params | Inference time/img |
|---|---|---|---|---|---|---|
| Baseline NAFNet | — | — | — | — | — | — |
| + Degradation-aware FiLM | — | — | — | — | — | — |
| + Log-domain transform | — | — | — | — | — | — |
| + Frequency branch | — | — | — | — | — | — |
| + Distilled student (final submission) | — | — | — | — | — | — |

---

## Part 6: How to Test — Practical Steps

### 6.1 Unit-level sanity checks (do these first, on your 3050, before any real training)

- Feed a single batch through the model and confirm output shape matches expected upscaled resolution.
- Confirm the degraded-image range (potentially outside [0,1]) is preserved through the input pipeline and not accidentally clipped.
- Overfit deliberately on a tiny subset (5–10 images) for many epochs — if the model can't drive loss near zero on a handful of examples, there's a bug in the architecture or loss, not a capacity issue.

### 6.2 Full training validation loop

- After each epoch (or every N steps), compute PSNR/SSIM/LPIPS on both in-distribution and pseudo-OOD validation splits.
- Visually inspect a handful of restored outputs vs. ground truth every few epochs — metrics can look fine while outputs have subtle artifacts (ringing, oversmoothing, color/intensity drift) that only visual inspection catches.
- Watch for divergence between in-distribution and pseudo-OOD metrics over training — growing divergence is your earliest overfitting signal, use it for early stopping.

### 6.3 Pre-submission testing (critical — this is what actually gets scored)

1. **Fresh-environment test**: clone your own GitHub repo into a brand-new environment (new Kaggle notebook or clean venv) and run your inference script exactly as a reviewer would — no manual path edits, no pre-loaded variables.
2. **Timing test on realistic batch size**: simulate the official benchmark by running inference on a full folder of test images and timing the entire script execution, not just the model forward pass.
3. **Edge case test**: run inference on a few visibly different/unusual images (if you have any OOD-style examples) to catch failure modes before the official test set does.
4. **Reproducibility test**: rerun your training script from scratch (or from a checkpoint) and confirm it reproduces comparable results — this is explicitly part of the "training hygiene" score.
5. **requirements.txt validation**: `pip install -r requirements.txt` into a clean venv and confirm no missing dependencies before submitting.

---

## Part 7: Summary — Why This Plan Addresses All Three Scored Axes

| Scored axis | How this plan addresses it |
|---|---|
| **Accuracy** (SSIM/PSNR/LPIPS, in-dist + OOD) | Degradation-aware conditioning, log-domain transform, and frequency branch each target a specific accuracy failure mode; ablation benchmarking proves their individual contribution |
| **Throughput** (H100 inference speed) | NAFBlock backbone, distillation to a smaller student, fp16 inference, and conv/BN fusion are all throughput-first design choices, benchmarked explicitly with end-to-end timing |
| **Training hygiene** | Checkpointed/resumable training, fixed seeds, ablation logging, fresh-environment reproducibility testing, and a standalone no-edit inference script directly satisfy this axis |

This plan is structured so every architectural and training decision maps back to a specific constraint stated in the problem — which is the strongest possible narrative for both the technical submission and the pitch deck.