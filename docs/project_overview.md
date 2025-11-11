# DynamicsNet Neural Longitudinal Dynamics Project

This document captures the design, implementation details, and evaluation outcomes of the neural replacement for the classical electric vehicle longitudinal dynamics model. It is intended to serve as a deep reference for developers, researchers, and operators working with this repository.

## Table of Contents

1. [Project Goals](#project-goals)
2. [System Architecture](#system-architecture)
   - [Data Pipeline](#data-pipeline)
   - [Model Suite](#model-suite)
   - [Training Regimen](#training-regimen)
   - [Evaluation & Visualization](#evaluation--visualization)
3. [Code Modules](#code-modules)
   - [Data Fetching](#data-fetching)
   - [Trip Parsing](#trip-parsing)
   - [Dataset Windows](#dataset-windows)
   - [Model Implementations](#model-implementations)
   - [Training Utilities](#training-utilities)
   - [Evaluation Tools](#evaluation-tools)
4. [Usage Guide](#usage-guide)
   - [Environment](#environment)
   - [Data Preparation](#data-preparation)
   - [Training](#training)
   - [Evaluation](#evaluation)
5. [Results & Diagnostics](#results--diagnostics)
   - [Aggregate Performance](#aggregate-performance)
   - [Horizon-Level Examples](#horizon-level-examples)
   - [Error Analysis](#error-analysis)
6. [Future Work](#future-work)

## Project Goals

- Replace the legacy physics-based EV longitudinal dynamics controller (`optimize_model_params_final.py`) with a data-driven solution that leverages transformers.
- Provide three cooperating neural modules:
  1. **Forward Model** – predicts future vehicle states given historical context and proposed actuations.
  2. **Inverse (Feedforward) Model** – predicts the required actuation sequence given desired future states.
  3. **Feedback Model** – generates residual corrections over the feedforward policy to compensate for discrepancies.
- Create a two-stage training pipeline: joint forward/inverse training followed by feedback fine-tuning with frozen counterparts.
- Offer comprehensive evaluation tooling, including visuals, to understand per-horizon performance, error distributions, and component contributions.

## System Architecture
### Transformer Internals

#### Input Token Construction

Each model constructs a token sequence

#### Dimensions & Segment Embeddings

All models use a shared embedding dimension `d_model` (default 128). Every token—regardless of whether it represents a state or action, history or horizon—gets projected into this space. Every projection outputs a tensor of shape `(batch, length, d_model)`. When we reference state–action fusion we mean the raw features are concatenated along the last dimension, projected once into `d_model`, and then segment embeddings are added. The projection itself produces the final `d_model`-sized token; there is no subsequent concatenation inside the transformer.

- **State projection** (`Linear(state_dim, d_model)`) and **action projection** (`Linear(action_dim, d_model)`) run on the concatenated state-action tensor because we first stack them along the feature dimension and only then project to `d_model`. The projection compresses the combined `(state_dim + action_dim)` vector into a single `d_model` token, so the resulting representation already embeds both modalities without requiring an additional concatenation step inside the transformer.
 `d_model` (default 128). Every token—regardless of whether it represents a state or action, history or horizon—gets projected into this space. Alignment is achieved with small linear adapters:

- **State projection** (`Linear(state_dim, d_model)`) and **action projection** (`Linear(action_dim, d_model)`) run on the concatenated state-action tensor because we first stack them along the feature dimension and only then project to `d_model`. The projection compresses the combined `(state_dim + action_dim)` vector into a single `d_model` token, so the resulting representation already embeds both modalities without requiring an additional concatenation step inside the transformer.

This parity means the attention layers operate on homogeneous token sizes; no specialized attention heads are required for different segments.

To distinguish semantic roles, we add **segment embeddings** `SegmentEmbedding(2, d_model)` before positional encoding. Index 0 is reserved for history tokens, index 1 for horizon tokens. During batching, each token receives:

```python
segment_ids = torch.zeros(batch, num_tokens, dtype=torch.long)
segment_ids[:, history_len:] = 1  # mark future segment
segment_embedding = segment_embed(segment_ids)
```

This additive bias informs the transformer that the first `H` tokens correspond to past observations/actuations, whereas the remaining `T` tokens correspond to future targets/plans. The attention mechanism remains fully bidirectional unless a mask is applied; thus horizon tokens can attend to history (for context) and to each other (for internal consistency), and history tokens can attend forward as well.

If a stricter separation is desired (e.g., horizon tokens should not influence history representations), one can supply a causal or block-diagonal mask when calling `TransformerBackbone`. In our current setup, this flexibility proved beneficial: feedforward predictions refine themselves by referencing historical states while still coordinating across future steps.

Padding is handled via key padding masks. Consider a batch where example `i` has effective history length `h_i <= H` and horizon length `t_i <= T`. The dataset emits masks `history_mask` and `future_mask` that indicate padded positions (1 = pad). `apply_masks(history_mask, future_mask)` stitches these into a single Boolean mask aligned with concatenated tokens. The transformer ignores padded positions both as queries and keys, ensuring variable-length support without reconfiguration.
 by concatenating history and horizon slices, mapping them into a shared embedding space. Let:

- `H`: history length
- `T`: horizon length
- `S_t`: state vector at time `t`
- `A_t`: action vector (throttle, brake) at time `t`

Context-specific constructions:

1. **Forward Model**
   - History tokens: `Embed_state(S_{t-H:t-1}) + Embed_action(A_{t-H:t-1})`.
   - Future tokens: `Embed_action(A_{t:t+T-1})` (candidate actions).

2. **Inverse Model**
   - History tokens: identical fusion (`state + action`), providing dynamics context.
   - Future tokens: `Embed_state(S^*_{t:t+T-1})` (desired states).

3. **Feedback Model**
   - History tokens: real states/actions plus residual history.
   - Future tokens: feedforward actions + desired states; residual head outputs adjustments.

Segment embeddings (`SegmentEmbedding`) encode whether a token belongs to history or horizon. Positional encoding (`PositionalEncoding`) preserves order within each segment.

#### Attention Masks

We use encoder-style Transformers, allowing full attention across history and horizon tokens. Attention masks (`key_padding_mask`) are driven by optional history/future padding flags, enabling variable-length windows without re-training. Causal masking is not applied by default (full bidirectional attention), which empirically improves horizon consistency by allowing future tokens to attend to history and vice versa.

- **History Mask**: indicates padded history entries (if any).
- **Future Mask**: indicates padded horizon entries (e.g., partial windows at trip tails).
- Combined via `apply_masks` (see `src/models/common.py`) into a single key padding mask fed to `TransformerBackbone`.

For scenarios requiring strict causality (e.g., onboard inference without peeking ahead), enable a triangular causal mask in the backbone.

#### Output Heads

- **Forward Model**: final token slice corresponding to horizon is projected back to state dimension (e.g., velocity, road grade predictions).
- **Inverse Model**: horizon slice passes through MLP head → `sigmoid` to bound `[0,1]` throttle/brake. Single-action representation (throttle positive, brake negative) can be derived downstream.
- **Feedback Model**: horizon slice outputs residuals in normalized actuation space; these are denormalized and added to feedforward actions before evaluation.

#### Combined Operation

At inference time (per window):

1. **Inverse (Feedforward) Inference**
   ```
   feedforward = inverse(history_states, history_actions, desired_future_states)
   feedforward = clamp(feedforward)
   ```
2. **Feedback Residual**
   ```
   residual = feedback(history_states, history_actions, desired_future_states, feedforward, history_residual)
   combined_actions = clamp(feedforward + residual)
   ```
3. **Forward Prediction**
   ```
   predicted_states = forward(history_states, history_actions, combined_actions)
   ```
4. Losses compare `combined_actions` vs. ground-truth actions and `predicted_states` vs. ground-truth states. During deployment, `predicted_states` can be fed into planning layers while `combined_actions` drives actuators.

The shared backbone parameters (hidden size, number of heads/layers) are controlled via `TransformerConfig`, promoting architectural consistency across modules.


### Data Pipeline

```
S3 Trip Buckets -> fetch_trips.py -> Raw Trip Folders
                -> parse_trips.py -> Time-aligned tensors (PyTorch .pt)
                -> EVSequenceDataset -> Training/eval windows (history + horizon)
```

1. **Fetching** – `scripts/fetch_trips.py` downloads trips filtered by car model, vehicle ID, date range, and required sensor files. Hardened to skip corrupt metadata while logging issues.
2. **Parsing** – `scripts/parse_trips.py` converts raw CSV sensor streams into synchronized arrays (speed, throttle, brake, road grade). Sanity checks detect degenerate timelines.
3. **Dataset** – `src/data/datasets.py` generates rolling windows of history (`history` steps) and prediction horizons (`horizon` steps) with on-the-fly normalization and validation.

### Model Suite

All transformer modules are built atop shared components from `src/models/common.py`:

- **PositionalEncoding** – sinusoidal embeddings for temporal ordering.
- **LinearProjection / SegmentEmbedding** – align state/action dimensionalities with the transformer embedding space.
- **TransformerBackbone** – configurable depth, heads, dropout, and normalization strategy.

Individual models:

1. **ForwardDynamicsModel (`src/models/forward.py`)**
   - Inputs: history states, history actions, candidate future actions.
   - Outputs: predicted future states (e.g., speed trajectory).
   - Architecture: Transformer encoder over concatenated history/future action tokens.

2. **InverseActuationModel (`src/models/inverse.py`)**
   - Inputs: history states/actions + desired future states.
   - Outputs: actuation sequence (bounded via `sigmoid` to [0, 1]).
   - Architecture: Similar to forward model but conditioned on desired states; produces feedforward policy.

3. **FeedbackResidualModel (`src/models/feedback.py`)**
   - Inputs: history states/actions, desired future states, feedforward actions, and historical residuals.
   - Outputs: residual actuation sequence.
   - Architecture: Transformer conditioned on both feedforward plan and desired targets.

### Training Regimen

Training resides in `src/training`:

1. **Stage 1 (`ForwardInverseTrainer`, `src/training/stage1.py`)**
   - Jointly trains forward and inverse models.
   - Loss terms:
     - `forward_loss` – L2 by default between predicted and ground-truth states.
     - `inverse_loss` – L2 between predicted and ground-truth future actions.
     - `consistency_loss` – encourages cyclic consistency between the two models.
   - Implements mixed precision, gradient clipping, and `tqdm` reporting of loss components.

2. **Stage 2 (`FeedbackTrainer`, `src/training/stage2.py`)**
   - Loads frozen forward/inverse weights from Stage 1.
   - Trains feedback model to predict residuals.
   - Loss terms:
     - `residual_loss` – residual actuations vs. actual corrections.
     - `tracking_loss` – optional forward rollout check using combined (feedforward + feedback) actions.
   - Also supports AMP and `tqdm` progress bars for training/validation.

### Evaluation & Visualization

`scripts/evaluate_longest_trip.py` orchestrates end-to-end inference on the longest validation trip segment:

- Loads Stage 1 and Stage 2 checkpoints.
- Replays the validation windows, collecting:
  - Per-step predictions (aggregate speed/actuation error tracking).
  - Horizon-level sequences for target vs. feedforward vs. feedback vs. combined outputs.
- Produces the following artefacts:
  - `plots/validation_longest_trip.png` – full-run overview.
  - `plots/validation_longest_trip_zoom.png` – zoomed horizon examples.
  - `plots/validation_longest_trip_diagnostics.png` – error distributions and scatter plots.
  - `plots/validation_longest_trip.npz` – serialized data for repeatable analysis.
  - `plots/validation_longest_trip.json` – summary statistics.

## Code Modules

### Data Fetching

- **Path**: `scripts/fetch_trips.py`, `src/data/fetch.py`
- **Highlights**:
  - Configurable destination (`data/raw/trips`).
  - Fault-tolerant JSON parsing; missing car metadata or corrupted files logged and skipped.
  - CLI arguments for car type, vehicle ID, date range, bucket, and optional dry runs.

### Trip Parsing

- **Path**: `scripts/parse_trips.py`, `src/data/parsing.py`
- **Workflow**:
  - Reads synchronized signals (wheel speeds, throttle, brake, road angle, driving mode).
  - Resamples onto a uniform timeline (`dt`) with interpolation.
  - Splits trips into segments (e.g., contiguous driving mode sequences).
  - Stores Tensor dictionaries keyed by segment ID with metadata summary.

### Dataset Windows

- **Path**: `src/data/datasets.py`
- **Features**:
  - Normalizes throttles/brakes to `[0, 1]` using `SequenceWindowConfig.max_throttle`/`max_brake`.
  - Provides composite actuation (positive throttle, negative brake) for downstream tasks.
  - Enforces validity masks (no NaNs, throttle/brake bounds, optional mutual exclusivity).
  - Returns `torch.Tensor` batches with keys: `history_states`, `history_actions`, `future_states`, `future_actions`, `future_actuation`.

### Model Implementations

- **`src/models/common.py`** – shared transformer scaffolding.
- **`src/models/forward.py`** – forward dynamics architecture, returning predicted states.
- **`src/models/inverse.py`** – inverse controller with `sigmoid` output head.
- **`src/models/feedback.py`** – residual model that consumes feedforward predictions and produces corrections.

All modules are exposed via `src/models/__init__.py` for concise imports.

### Training Utilities

- **`src/training/losses.py`** – L1/L2/SmoothL1 regression loss configuration plus cyclic consistency implementation.
- **`src/training/utils.py`** – helpers for device transfer and running average metrics.
- **`src/training/stage1.py`** – joint trainer with `ForwardInverseTrainingConfig`.
- **`src/training/stage2.py`** – feedback trainer with `FeedbackTrainingConfig`.

Both trainers rely on `torch.cuda.amp` with graceful degradation on CPU fallback, and print compact metrics via `tqdm.set_postfix`.

### Evaluation Tools

- **`scripts/evaluate_longest_trip.py`** – core evaluation CLI (documented below).
- Captures per-step and per-window results, writes summary JSON and `.npz` for reproducibility.
- Generates multi-figure plots (see Results section).

## Usage Guide

### Environment

```bash
# Ensure you are inside the repo root
cd /opt/imagry/DynamicsNet

# (Optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (PyTorch, numpy, pandas, tqdm, matplotlib, boto3, etc.)
pip install -r requirements.txt
```

### Data Preparation

1. **Fetch trips** (example – adjust date range as needed):

```bash
PYTHONPATH=. python scripts/fetch_trips.py \
  --car ECentro \
  --vehicle-id ECENTRO_HA_02 \
  --start 2025-07-01 \
  --end 2025-10-31
```

2. **Parse trips** into processed tensors:

```bash
PYTHONPATH=. python scripts/parse_trips.py \
  --input data/raw/trips \
  --output data/processed/ECentro/ECENTRO_HA_02/all_trips_data.pt
```

### Training

1. **Stage 1 (forward + inverse)**:

```bash
PYTHONPATH=. python scripts/train_stage1.py \
  --dataset data/processed/ECentro/ECENTRO_HA_03/all_trips_data.pt \
  --batch-size 512 \
  --epochs 50 \
  --val-share 0.1 \
  --out-dir checkpoints/stage1/ecentro_ha_03_bs512
```

2. **Stage 2 (feedback residual)**:

```bash
PYTHONPATH=. python scripts/train_feedback.py \
  --dataset data/processed/ECentro/ECENTRO_HA_03/all_trips_data.pt \
  --stage1-checkpoint checkpoints/stage1/ecentro_ha_03_bs512/stage1.pth \
  --batch-size 512 \
  --epochs 100 \
  --val-share 0.1 \
  --out-dir checkpoints/stage2/ecentro_ha_03_bs512
```

Training scripts automatically handle gradient scaling, gradient clipping, checkpointing, and progress reporting.

### Evaluation

Run the consolidated evaluation pipeline:

```bash
PYTHONPATH=. python scripts/evaluate_longest_trip.py \
  --dataset data/processed/ECentro/ECENTRO_HA_03/all_trips_data.pt \
  --stage1 checkpoints/stage1/ecentro_ha_03_bs512/stage1.pth \
  --stage2 checkpoints/stage2/ecentro_ha_03_bs512/stage2.pth \
  --output plots/validation_longest_trip.png
```

Outputs:

- `plots/validation_longest_trip.png`
- `plots/validation_longest_trip_zoom.png`
- `plots/validation_longest_trip_diagnostics.png`
- `plots/validation_longest_trip.npz`
- `plots/validation_longest_trip.json`

## Results & Diagnostics

### Aggregate Performance

![Full Run](../plots/validation_longest_trip.png)

- **Speed RMSE**: ~0.23 m/s (R² ≈ 0.9955) – strong state-tracking fidelity.
- **Throttle RMSE**: ~5.98% (R² ≈ 0.886) – greater variance due to sharp actuation changes.
- **Brake RMSE**: ~2.65% (R² ≈ 0.88) – residual controller effectively reduces error.

The overview plot shows GT vs. predicted speed, aggregated throttle/brake commands, and decomposed feedforward/feedback contributions.

### Horizon-Level Examples

![Zoomed Horizons](../plots/validation_longest_trip_zoom.png)

- Rows represent speed, throttle, and brake trajectories across the prediction horizon for four representative windows.
- Curves illustrate how feedforward actions relate to GT actuations and how feedback residuals adjust the plan.
- Highlights the controller’s responsiveness to varying terrain and speed demands.

### Error Analysis

![Error Diagnostics](../plots/validation_longest_trip_diagnostics.png)

- **Histograms** – reveal skewness and heavy tails, particularly for throttle where large corrections occur.
- **Scatter Plots** – GT vs. final predicted actuations show strong correlation with some systematic bias in high throttle regimes.
- **MAE vs. Horizon Step** – quantifies where predictions degrade across the horizon. Speed remains stable; throttle/brake drift slightly at longer horizons, suggesting opportunities for adaptive weighting or horizon-specific training tweaks.

### Supplementary Data

- `validation_longest_trip.npz` contains the serialized arrays used to create these plots (per-window targets, predictions, feedforward/feedback components, etc.).
- `validation_longest_trip.json` logs the scalar metrics for quick reference or regression tracking.

## Future Work

1. **Model Improvements**
   - Explore hybrid loss functions (e.g., SmoothL1 or heteroscedastic variance modeling) to better handle throttle outliers.
   - Incorporate additional context (battery SOC, ambient conditions) if available to the dataset.
   - Evaluate causal mask variants to enforce strict autoregressive behaviour when desired.

2. **Training Enhancements**
   - Curriculum schedules that gradually increase horizon length.
   - Data augmentation techniques (e.g., noise injection, synthetic slopes) to broaden generalization.
   - Automated hyper-parameter sweeps (optimizer, dropout, transformer depth).

3. **Evaluation & Deployment**
   - Online (closed-loop) evaluation within a simulation stack.
   - Export scripts for ONNX or TensorRT deployment.
   - Integration with real-time telemetry dashboards using saved diagnostics.

## References

- `src/` – core library (`data`, `models`, `training`).
- `scripts/` – top-level CLIs for fetch, parse, train, evaluate.
- `plots/` – generated artefacts for the current dataset snapshot.
- `docs/` – documentation (including this file).

For any additions or modifications, ensure new experiments append their outputs within `plots/` or subdirectories to keep history consistent.

