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

## Status

- [x] Stock EAGLE global regression passes (2026-10-06, `run/global-regtest-20261006T1438`).
- [x] `config/global.yaml` ported to anemoi-training 0.17; global regression passes on
      `anemoi-dev` (`run/global-regtest-dev-20261006T1617`); dev env use verified from
      checkpoint provenance.
- [x] aifs2-1deg in EAGLE, Bo's settings (`multistep_input: 2`): 1-GPU smoke test passed
      (job 23179565: 231 M params, 5.3 it/s, checkpoint 935.19 MB, graph built in-job).
- [x] Switch to a single initial condition (`app.time.multistep_input: 1`): smoke passed
      (job 23179679: 230 M params, 5.5 it/s).
- [ ] aifs2-1deg-ic1 full pretraining, job 23180416 (8 nodes x 2 H100, 40 h limit; Bo's 2-IC run
      took 21h41m). Run dir `run/aifs2-1deg-ic1/training`. Bo's 2-IC config was smoke-tested only.
- [ ] Fine-tuning stage (rollout to 12, 13 epochs; Bo's took 22h43m).
- [ ] Inference + verification for aifs2 (ERA5 variable names differ from the GFS vx config).
- [ ] Portability: env spec, site-specific settings, data transfer; push branches.

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
