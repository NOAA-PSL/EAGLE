# latent-rollout: plan and status

Working notes for the `latent-rollout` branch. Keep **Status** current and append to **Log**
(newest last) whenever something is decided, run, or verified.

## Goal

1. Run EAGLE on the latest anemoi, using the NOAA-PSL forks of anemoi-core and anemoi-inference
   (branch `feature/latent-rollout-plan`, which adds latent-space autoregressive rollout; see
   `plan_latent_rollout.md` in each fork).
2. Inside EAGLE, recreate Bo Huang's AIFS v2 1-degree (O96) ERA5 run
   (`/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/aifs-single-mse-2.0-1deg`, anemoi-training 0.6.3),
   then change it to a single initial condition (`multistep_input: 1`).
3. Make the setup reproducible on another HPC.

## Layout (Ursa)

All under `/scratch4/BMC/gsienkf/Sergey.Frolov/eagle/latent/`:

| Path | What |
|---|---|
| `EAGLE/` | this repo, branch `latent-rollout`; conda envs in `EAGLE/conda/` |
| `anemoi-core/`, `anemoi-inference/` | NOAA-PSL forks, `feature/latent-rollout-plan` (core `7b53fcb`, inference `2169c56`) |
| `eagle-tools/` | NOAA-PSL/eagle-tools, local branch `latent-rollout` off v0.8.1 (inference fix, below) |
| conda env `anemoi-dev` | clone of `anemoi` + editable installs of the forks and eagle-tools; rebuild with `../make-anemoi-dev` |
| `aifs2-1deg/` | earlier standalone (non-EAGLE) port of Bo's config; reference only |

## How to run

```bash
# Global GFS regression on the dev env (end to end):
ANEMOI_ENV=anemoi-dev ../global-eagle-regression

# aifs2 1-degree: compose the case, set app.base/experiment_name/account, then train.
make config compose=base:aifs2-1deg:ursa:anemoi-dev[:aifs2-1deg-smoke] >eagle.yaml
make training config=eagle.yaml ANEMOI_ENV=anemoi-dev   # no `make data`: reads Bo's ERA5 zarrs
```

## Decisions

- **Dev env**: `anemoi-dev` pins torch 2.10 / flash-attn 2.8.3 / triton / numpy from the stock env;
  other deps follow the forks (e.g. anemoi-transform 0.1.29 -> 0.5.0).
- **Makefile `ANEMOI_ENV`**: login-node steps (training config generation, checkpoint validation)
  must run in the same env as the jobs, or configs are built against the old anemoi.
- **GPU jobs**: account `gpu-ai4wp`, QoS `gpu` (not `gpuwf`).
- **`val.gpus_per_node`** replaces the hard-coded `SLURM_GPUS_PER_NODE=1`, for multi-node training.
- **eagle-tools fix**: anemoi-inference >= 0.12 rejects reusing an output path in one process;
  eagle-tools' throwaway model-loading runner now uses output `none`. Also broken upstream (v0.9.0).
- **aifs2 data**: read Bo's ERA5 zarrs in place (main v8 zarr is public at
  `https://data.ecmwf.int/anemoi-datasets/era5-o96-1979-2023-6h-v8.zarr`; the 10 hPa supplement
  was built by Bo from CDS).
- **aifs2 graph**: built by the training job (as in the GFS cases). Ursa GPU nodes have no
  internet, so the o48 grid must be in `~/.cache/anemoi/grids` (fetch once on a login node).
- **Nested EAGLE**: not ported (global only).

## Latent-rollout training plan (1 IC)

Prerequisite: the forks' Phases 1-2 (`AnemoiModelLatentRollout` with `encode`/`process`/`decode`,
and a `LatentForecaster` task). As of 2026-10-06 the fork branches contain only the plan document.

### Lessons from published latent-rollout models

- **WeatherMesh-3** (WindBorne, ICLR 2025 workshop, arXiv:2503.22235): 6 h latent processor trained
  first, with the target horizon extended in step-count stages (0-12 h for steps 0-1k, 0-24 h to 15k,
  0-36 h to 21k, ... up to 120 h = 20 latent steps); then a 1 h processor trained with the
  encoder/decoder **frozen** (25k steps, random dt in 0-24 h). Loss: MSE on decoded output at the
  target dt; no latent-space loss and no explicit anti-drift mechanism described.
- **GraphDOP / AI-DOP** (ECMWF, arXiv:2412.15687; coupled version arXiv:2510.20416): observations in,
  observations out; latent stepped every 3 h across a 12 h window. Single-window pretraining to step
  63k, then **+1 target window every 1.5k steps**. Weighted MSE per instrument/channel; random
  observation dropout (1/4 satellite, 1/2 conventional) as regularization.
- Takeaways: (1) ramp rollout by **step count in small increments**, not +1 per epoch (Bo's grid
  fine-tune adds 1 per ~3.9k-step epoch); (2) training new parts against a **frozen**
  encoder/decoder works; (3) both reach stable long latent rollouts with decoded loss only, so a
  latent-consistency loss is an **ablation**, not a default.

### Why latent rollout may drift more than grid rollout

Grid rollout re-projects every step (decode -> bound -> normalize -> re-encode), discarding latent
components the decoder ignores and keeping the encoder on physical-looking inputs. Latent rollout
loses that projection: errors can grow in decoder-invisible directions, the processor sees its own
outputs (exposure bias), grid-space constraints (bounding, skip connection) are absent, and
forcings enter through a new learned embedder. In exchange it avoids the per-step information loss
(blurring). Mitigations, in order: earlier/longer multi-step training; latent consistency loss
(`process^k(encode(x_t)) ~ sg(encode(x_t+k))`); normalizing `z` between steps.

### Curriculum

| Stage | Trains | Rollout | Length (rough) | LR (local) | Gate to pass |
|---|---|---|---|---|---|
| 0 Baselines | grid model, 1 IC: `aifs2-1deg-ic1` pretrain + Bo-style grid fine-tune | 1, then 1->12 | ~20 h + ~23 h | Bo's | - |
| 1 Adapt | latent model warm-started from stage 0; freeze encoder+processor (`submodules_to_freeze`), train forcing embedder + decoder (no `x_skip`), then unfreeze | 1 | ~10-20k steps | ~1e-5 | 1-step RMSE = stage 0 within noise |
| 2 Ramp | all weights; decoded loss every step; checkpoint each latent step | 2->4->8->12, stages every few k steps | ~1 day | ~3e-6 decaying | each length stable before doubling |
| 3 Long | all weights | 12 (3 d), optionally 20 (5 d) | ~1 day | 8e-7 | 5-10 d RMSE >= stage 0 grid fine-tune |

Ablations: latent consistency loss; LayerNorm on `z`; lead-time loss weighting; a WeatherMesh-style
second short-dt processor trained against the frozen encoder/decoder.

### Reusable anemoi code (checked 2026-10-06)

ECMWF's public DOP work in anemoi-core is on `upstream/cherri-dop-native-observations` (317 commits,
active to 2026-09-23): native/tabular observation data sources, batch layouts, multi-encoder
latent fusion and sharding. It does **not** contain a multi-step latent processor loop (one
`self.processor(...)` call per forward); GraphDOP's in-window latent stepping is not public there.
Related: `feat/latent_fusion`, `feat_multi_encoder_latent_comb_2` (multi-encoder latent combining),
`feat/encoded-latents` (decoder targets from encoded latents). Already on main: the `Autoencoder`
task (`task: autoencoder`), useful for stage 1-style encoder/decoder work.

## Status

- [x] Stock EAGLE global regression passes (2026-10-06, `run/global-regtest-20261006T1438`).
- [x] `config/global.yaml` ported to anemoi-training 0.17; global regression passes on
      `anemoi-dev` (`run/global-regtest-dev-20261006T1617`); dev env use verified from
      checkpoint provenance.
- [x] aifs2-1deg in EAGLE, Bo's settings (`multistep_input: 2`): 1-GPU smoke test passed
      (job 23179565: 231 M params, 5.3 it/s, checkpoint 935.19 MB, graph built in-job).
- [x] Switch to a single initial condition (`app.time.multistep_input: 1`): smoke passed
      (job 23179679: 230 M params, 5.5 it/s).
- [x] aifs2-1deg-ic1 full pretraining, job 23180416: completed in 21h46m (260k steps, 66 epochs).
      Checkpoint `run/aifs2-1deg-ic1/training/outputs/checkpoint/8a8641f4213949afb3c68365ea7d1e63/`.
      vs Bo's 2-IC at epoch 66 (6 h val MSE ratio ours/Bo): loss 1.04, z500 1.09, msl 1.07, t850 1.03,
      winds/2t 1.01-1.02, tp 1.14 (tp val error rises after ~epoch 40 in both runs).
- [ ] Compare `aifs2-1deg-ic1` (1 IC) against Bo's 2-IC pretraining checkpoint (no 2-IC rerun in
      EAGLE, by decision): `/scratch4/BMC/gsienkf/Bo.Huang/expCodes/ML/aifs-single-mse-2.0-1deg/outputs/
      pretraining/checkpoint/1166d230b93141a8bfe60d43841ba7b3/inference-anemoi-by_epoch-epoch_066-step_260000.ckpt`.
- [ ] Stage 0 grid fine-tune of `aifs2-1deg-ic1` (rollout to 12, 13 epochs; Bo's took 22h43m).
- [ ] Inference + verification for aifs2 (ERA5 variable names differ from the GFS vx config).
- [ ] Portability: env spec, site-specific settings, data transfer.
- [ ] Push: not a NOAA-EPIC member (can't push upstream); eagle-tools is read-only (access requested).

## Log

- 2026-10-06: Built stock envs; global regression passed on stock anemoi.
- 2026-10-06: Created `anemoi-dev`; ported global training config (task group, optimization,
  bounding, plot callbacks, system.input); fixed eagle-tools inference; dev regression passed.
- 2026-10-06: Ported Bo's pretraining config standalone (`aifs2-1deg/`); matched graph
  (4,027,434 vs 4,027,498 bytes) and checkpoint size (935.2 vs 933.7 MB, 231 M params);
  1-GPU smoke passed (job 23175516). User then asked to redo it inside EAGLE.
- 2026-10-06: Added `config/aifs2-1deg.yaml` + `aifs2-1deg-smoke.yaml`; resolved training config
  matches the standalone port. Smoke job 23179565 passed.
- 2026-10-06: Single initial condition (`multistep_input: 1`, Bo used 2): smoke passed;
  submitted full pretraining `aifs2-1deg-ic1` as job 23180416.
- 2026-10-06: Job 23180416 running at ~3.6 it/s (~20 h), effective LR 5e-4. Decided to compare
  against Bo's 2-IC pretraining checkpoint rather than rerun 2-IC in EAGLE.
- 2026-10-06: Added latent-rollout training plan (WeatherMesh-3 and GraphDOP lessons, curriculum,
  reusable upstream DOP branches).
- 2026-10-07: 1-IC pretraining finished; val curves vs Bo in `../figures/aifs2-compare/`
  (1 IC ~3-9% worse on mass fields, 14% on tp; winds/temps within noise).
