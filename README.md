# ReSpecRank

ReSpecRank is a PyTorch package for regime-adaptive joint time-graph spectral filtering
and cross-sectional stock ranking. The repository includes local-data preprocessing,
training, evaluation, diagnostics, configuration, and tests. It contains no market-data
downloader and does not redistribute data.

## What is implemented

- A daily point-in-time universe with variable stock count and no batch padding.
- Twelve price-volume features over 20 input dates, daily 1%/99% winsorization, and
  cross-sectional standardization.
- Five-day relative excess log-return targets with daily standardization.
- A trailing 60-return positive-correlation graph, top-10 neighbors, symmetrization,
  normalized Laplacian, and sparse scaled-Laplacian edges.
- A shared `12 -> 128` feature projection.
- Chebyshev graph orders `0, 1, 2, 3` and temporal dilations `0, 1, 2, 4, 8, 16`.
- Two stacked joint filtering layers, each with four learned `4 x 6` coefficient surfaces.
- A bias-free `128 x 128` channel-mixing matrix for every graph-temporal branch in
  every joint filtering layer.
- The `5 -> 16 -> 4` GELU/softmax regime router with a zero-initialized output layer.
- Uniform routing during the first five epochs, followed by adaptive routing.
- Per-layer residual fusion and layer normalization, followed by the
  `128 -> 128 -> 1` GELU/dropout ranking head.
- The sampled pairwise logistic loss, standardized-score Huber loss, and 32-by-32
  joint-response diversity penalty.
- RankIC, IC, ICIR, response centroids, state shuffling, paired moving-block intervals,
  and blockwise sign-flip tests.
- Checkpointing, train/validation selection, deterministic seeds, ablation configuration,
  and a synthetic end-to-end smoke-test path.

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

PyTorch installation can be adjusted for the local CUDA environment before installing
the package. No command in this repository downloads stock data.

## Local input format

`prices.csv` is long-form and must contain already adjusted, point-in-time values:

| Column | Meaning |
|---|---|
| `date` | Trading date |
| `symbol` | Stable security identifier |
| `open`, `high`, `low`, `close` | Adjusted OHLC prices |
| `volume` | Adjusted or consistently defined volume |
| `turnover` | Turnover feature supplied by the data owner |
| `tradable` | Boolean; optional, defaults to true |

`membership.csv` contains a complete constituent snapshot for each prediction date:

| Column | Meaning |
|---|---|
| `date` | Snapshot effective date |
| `symbol` | Constituent identifier |
| `is_member` | Boolean; optional when every row is active |

Do not backfill membership snapshots. Input rows should already reflect adjustment factors
known under the chosen data protocol. A missing or non-tradable quote causes the
stock-date to be excluded; it is never filled with zero.

Map alternative column names in `configs/prepare_example.yaml`, then prepare the dataset:

```bash
respecrank-prepare --config configs/prepare_example.yaml
```

The output has one compressed NumPy file per date and a manifest. Market-state statistics
are normalized only with training-split moments. The train, validation, and test universes
remain unpadded.

## Training and evaluation

```bash
respecrank-train --config configs/respecrank.yaml
respecrank-evaluate \
  --checkpoint outputs/respecrank/best.pt \
  --split test
respecrank-diagnose \
  --checkpoint outputs/respecrank/best.pt \
  --seed 42
```

The default training configuration uses 80 epochs, eight complete dates per batch, AdamW,
learning rates `1e-3` for the main network and `3e-4` for the router, 4096 sampled pairs
per date, and five uniform-routing warm-up epochs. Early stopping is disabled by default.
`best.pt` is selected by validation RankIC; `last.pt` is also retained.

The suggested five-seed evaluation set is:

```text
1, 7, 42, 2025, 2026
```

Run each seed in a separate output directory. Aggregation is intentionally not hidden in
the training command so every run has an auditable resolved configuration and history.

## Synthetic pipeline check

The synthetic generator exists only to verify pipeline mechanics and software behavior.

```bash
python scripts/make_synthetic_data.py
respecrank-prepare --config configs/synthetic_prepare.yaml
respecrank-train --config configs/synthetic_experiment.yaml
respecrank-evaluate --checkpoint outputs/synthetic/best.pt --split test
```

## Component map

| Component | Implementation |
|---|---|
| Target construction | `preprocessing.py::_build_item` |
| Graph construction and sparse operator | `graph.py` |
| Stacked joint filter | `model.py::ReSpecRank.forward` |
| Joint transfer response | `losses.py::joint_response` |
| Market-state construction | `preprocessing.py::_market_statistics` |
| Router and coefficient composition | `model.py::RegimeRouter` and `composite_coefficients` |
| Training objectives | `losses.py` |
| Response centroids | `diagnostics.py` |

## Ablations and controls

The principal ablations require only configuration changes:

- **FixedJoint:** set `model.routing_mode: uniform`.
- **Time-only:** set `model.graph_order: 0`.
- **Graph-only:** set `model.temporal_dilations: [0]`.
- **Without diversity:** set `loss.diversity_weight: 0.0`.
- **Single basis:** set `model.num_bases: 1`.
- **State shuffle:** run `respecrank-diagnose`; model weights and price/graph inputs stay
  fixed while test-date market states are permuted.

The data builder produces dynamic daily graphs. A static-graph control should be created
as a separate processed dataset so its graph provenance is explicit rather than silently
changing behavior inside the model.

## Implementation conventions

The following data and model conventions are explicit and configurable:

1. Returns are log close-to-close returns. Five-day momentum is the simple five-day close
   return. Relative OHLC features are `price / close - 1`, volume change is a log change,
   and realized volatility uses population standard deviation.
2. The low-frequency energy set defaults to real-FFT bins `[0, 1, 2]` over the 20-date
   market-return window. Change `low_frequency_bins` as an explicit configuration choice.
3. Boundary purging drops the last five dates of
   both the training and validation sections, preventing five-day targets from crossing a
   split boundary.
4. Training runs all 80 epochs unless `early_stopping_patience` is explicitly configured.
   The best validation checkpoint and final checkpoint are both retained.
5. The default model uses two filtering layers with branch-specific `128 x 128` channel
   maps:

   ```text
   input projection                         1,664
   2 x 24 x 128 x 128 channel maps        786,432
   2 x 4 x 24 spectral coefficients           192
   shared router                               164
   two LayerNorm modules                       512
   ranking head                             16,641
   total                                   805,605
   ```

   Channel maps are bias-free. Enabling one bias vector per branch would add 6,144
   parameters and produce 811,749 parameters.

6. Stacked temporal shifts use a causal left boundary: unavailable history before the
   20-date input window contributes zero. Every layer computes the complete sequence, so
   layer `l` reads delayed representations produced by layer `l-1`. The current-date graph
   is applied to all sequence positions.

7. Response diagnostics summarize the root-mean-square magnitude of the two scalar
   coefficient surfaces. They do not claim to be the exact matrix-valued transfer response
   after branch-specific channel mixing.

These choices should be held fixed across compared methods. Configuration changes should
be recorded with the generated run metadata.

## Tests

```bash
pytest
ruff check src tests scripts
```

Tests cover graph construction, sparse Chebyshev recurrence, router initialization, model
forward/backward behavior, spectral losses, metrics, processed-item loading, and
checkpoint reconstruction.
