# Diffolio

A PyTorch implementation of **Diffolio** from *Diffusion Models for Risk-Aware
Portfolio Optimization* (Jihyeong Jeon, Jeongyoung Lee, U Kang, WSDM '26).

Diffolio skips forecasting prices. Given a window of market history and a risk
level chosen by the user, it samples portfolio weights directly from a
conditional diffusion model. The weights are trained to match "pseudo-optimal"
portfolios built from realised returns, and a risk-guidance term pushes each
reverse step towards lower portfolio variance.

## How it works

**Data.** The universe is a fixed, ordered set of `N` assets plus a benchmark
index. Daily OHLCV bars are turned into `F` normalised features per asset. At
each decision step `τ` the model sees a look-back window `h ∈ R^{N×L×F}` of
asset history and `g ∈ R^{L×F}` of index history. The target is the next-day
open-to-open return `r_τ ∈ R^N`.

**Pseudo-optimal portfolios.** In hindsight the best allocation leans towards
the assets that moved most. The paper turns this into targets with two
L1 normalisers:

- `f(r) = r / Σ|r|` gives the base portfolio `x^(τ)` (Eq. 12).
- `f_γ` keeps only the top-`k_γ` assets by `|r|` before normalising (Eqs.
  15–16). `k_γ = ⌊N/γ_max⌋·(γ_max − γ)`, so `γ = 0` spreads weight widely and
  `γ = γ_max − 1` concentrates it.

The sign of each return carries through, so short positions are allowed.

**Diffusion.** The model is a standard DDPM, except that the noise is scaled to
the data. Both the forward process and the prior use `N(0, σ_x² I)`, where
`σ_x` is the spread of the base portfolios over the training period.

**Denoising network `F_γ`.**
- *Market encoder.* Dilated TCNs produce per-asset queries, keys and values.
  Self-attention across assets follows, then a second TCN block. The index
  goes through an LSTM with temporal attention. The two are fused per asset
  and attention-pooled into `z_merged ∈ R^d`, with `d = L·F`.
- *Denoising head.* The noisy portfolio `x_t` is projected to `R^d` and
  modulated by a time embedding plus a learned risk embedding `v_γ` (Eq. 17).
  It is then concatenated with `z_merged`, and an MLP predicts the clean
  portfolio `x̂_0`.

**Training** (Algorithm 1). Minimise the denoising MSE against `x_0^(τ,γ)`,
minus `λ` times an auxiliary return objective, `⟨f(W_p z_merged), r_τ⟩`
(Eq. 18). The auxiliary term makes the encoder learn return-relevant features
on its own. `τ`, `γ` and `t` are sampled uniformly for each example.

**Sampling with risk guidance** (Algorithm 2). At each reverse step the proxy
risk `ρ' = wᵀ Σ̂ w`, with `w = f(x_t)`, is differentiated with respect to
`x_t`. Its gradient, weighted by `ζ_γ = 1 − γ/(γ_max − 1)`, is subtracted from
the posterior mean, so low risk levels are pulled harder towards
low-variance allocations. The final sample goes through `f_γ` to produce the
weights.

**Evaluation.** The model is backtested on the held-out test period using ARR,
ASR, MDD, AVol, Calmar and Sortino ratios, averaged over risk levels and
sampling seeds.

## Implementation details not specified in paper

The paper does not specify the data pipeline, so this implementation supplies
one:

- **Universe.** Candidates come from an index roster (S&P 500 for the U.S.
  config) and are downloaded, then screened on minimum price, *median* daily
  dollar volume, missing data and continuous listing. The ordered list of `N`
  survivors is frozen only after screening, and every tensor's asset axis
  follows that order.
- **Prices.** Data comes from `yfinance` with `auto_adjust=True`. Returns are
  open-to-open, so an unadjusted open would create a fake gap on every split
  and ex-dividend date.
- **Features.** `F = 5` (OHLCV). Prices are divided by a trailing reference
  price and volume is log-relative, so the transform is causal. An optional
  z-score is fitted on the training split only.
- **Gaps.** Missing bars are forward-filled, and every filled cell is recorded
  in an `observed` mask. A return with a filled leg is marked invalid. A
  decision step is dropped when too much of its window or target is filled
  (`split.max_window_fill_frac`, `split.min_valid_target_frac`). Assets with a
  leading gap are dropped instead of back-filled, because back-filling would
  copy future prices into the past.
- **Splits.** The split is chronological, 7:1:2. A step belongs to a split
  only if its whole window and the next day's return fall inside it.
- **Targets.** Top-`k_γ` selection never picks an asset whose return is
  invalid. Ties break towards the lower asset index, so targets are
  deterministic.

Where the paper's details conflict or are missing:

- **MLP output width.** The paper writes the head as `R^{2d} → R^d`, but its
  output is a portfolio in `R^N`. The final layer projects to `N`.
- **Time-embedding nonlinearity.** The paper's `σ` is read as a sigmoid
  (`model.time_embedding_activation`).
- **Reverse-step noise.** Algorithm 2 draws `z ~ N(0, σ_x² I)` and then
  multiplies by `σ_x` again. That contradicts the stated reverse covariance,
  so `σ_x` is applied only once.
- **Covariance `Σ̂`.** The paper says only "estimated from historical returns".
  The default here is a trailing window ending at `τ`, with shrinkage, because
  `N` can exceed `L`. A single covariance frozen on the training data is the
  alternative.
- **AVol** is annualised by `√D_Y`, not `D_Y` as the paper's text reads.
  Table 4's magnitudes only match `√D_Y`.

## Layout

```
src/diffolio/
  config.py       DiffolioConfig: one YAML-backed config; d = L·F is derived, not stored
  portfolio.py    f, f_γ and k_γ, shared by targets, the auxiliary loss and sampling
  data/
    universe.py   roster → screened, ordered universe
    download.py   yfinance acquisition with a parquet cache
    clean.py      calendar alignment, gap tracking, anomaly checks
    features.py   causal per-asset normalisation
    splits.py     chronological split and usable decision steps
    panel.py      MarketPanel: aligned (T, N, F) / (T, F) / (T, N) arrays
    windows.py    WindowDataset: (h, g, r) samples sliced from the panel
    targets.py    base and risk-dependent pseudo-optimal portfolios
    pipeline.py   build_dataset: the end-to-end, fingerprint-cached build
scripts/build_dataset.py
configs/us_sp500.yaml
```

Windows are sliced from the memory-mapped panel on demand instead of being
materialised. Each day appears in `L` overlapping windows, so a stored copy
would be roughly `L` times the size of the panel.

## Usage

```bash
pip install -r requirements.txt && pip install -e .
python scripts/build_dataset.py --config configs/us_sp500.yaml \
    [--set universe.target_size=50] [--force]
```

The output goes to `data/processed/<name>/`, and raw downloads are cached in
`data/cache/`. The build is keyed on a fingerprint of the data-related config
sections, so changing `model` or `training` settings does not trigger a
rebuild.

```python
from torch.utils.data import DataLoader
from diffolio.config import DiffolioConfig
from diffolio.data import build_dataset, build_targets, portfolio_dataset

config = DiffolioConfig.from_yaml("configs/us_sp500.yaml")
dataset = build_dataset(config)
train = portfolio_dataset(dataset, "train", build_targets(dataset))
batch = next(iter(DataLoader(train, batch_size=128, shuffle=True)))
# batch.h (B,N,L,F)  batch.g (B,L,F)  batch.r (B,N)  batch.x_base (B,N)  batch.x0 (B,γ_max,N)
```
