"""Write EAGLE/config/aifs37-a{0,1,2}.yaml (usage: python gen_aifs37.py EAGLE/config) (the per-run level sets of the AIFS 37-level round 1)."""
import sys
OUT = sys.argv[1]
L37 = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550,
       600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
RUNS = {
    "a0": ([10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], "Bo's 14 levels, 10 hPa top"),
    "a1": (L37, "all 37 ERA5 levels, 1 hPa top"),
    "a2": ([p for p in L37 if p >= 10], "P32: the 37 levels below 10 hPa, 10 hPa top"),
}
SFC = ["10u", "10v", "2d", "2t", "cos_julian_day", "cos_latitude", "cos_local_time", "cos_longitude", "cp",
       "insolation", "lsm", "msl", "sdor", "sin_julian_day", "sin_latitude", "sin_local_time", "sin_longitude",
       "skt", "slor", "sp", "tcw", "tp", "z"]
PRECURSORS = ["sd", "siconc", "swvl1", "cvh", "cvl", "lai_hv", "lai_lv", "slt", "fsr"]


def flow(xs):
    return "[" + ", ".join(xs) + "]"


for run, (levels, what) in RUNS.items():
    pl = [f"{v}_{p}" for v in ["q", "t", "u", "v", "w", "z"] for p in levels]
    q = [f"q_{p}" for p in levels]
    qdiag = [f"q_{p}" for p in levels if p <= 50]  # as Bo: stratospheric humidity is output only
    w = [f"w_{p}" for p in levels]
    text = f"""# AIFS 37-level recipe, round 1, run {run.upper()}: {what}. Compose after aifs37, e.g.
#   make config compose=base:aifs2-1deg:aifs37:aifs37-{run}:ursa:anemoi-dev[:mi300x]
# Written by gen_aifs37.py (lists are long); all three runs take their pressure levels from the 37-level
# zarr, so they differ only in the levels.

# yamllint disable rule:line-length

app:
  experiment_name: aifs37-{run}
training:
  anemoi:
    data:
      datasets:
        data:
          diagnostic: {flow(["tp", "cp"] + qdiag + w)}
          processors:
            normalizer:
              config:
                std: {flow(["tp", "cp", "tcw", "sd", "swvl1"] + q)}
    model:
      bounding:
        datasets:
          data:
            - _target_: anemoi.models.layers.bounding.ReluBounding
              variables: {flow(["tp", "tcw", "sd", "swvl1"] + q)}
            - _target_: anemoi.models.layers.bounding.HardtanhBounding
              variables: [siconc]
              min_val: 0
              max_val: 1
            - _target_: anemoi.models.layers.bounding.FractionBounding
              variables: [cp]
              min_val: 0
              max_val: 1
              total_var: tp
            - _target_: anemoi.models.layers.bounding.GlobalDryMassFixer
              variables: [sp]
              water_var: tcw
val:
  era5:
    dataset:
      join:
        - dataset: '{{{{ val.era5.dir }}}}/era5-o96-1979-2023-6h-v8.zarr'
          select: {flow(SFC)}
          start: '{{{{ val.era5.start }}}}' # join needs equal dates: v8 cut to the new zarrs' range
          end: '{{{{ val.era5.end }}}}'
        - dataset: '{{{{ val.era5.dir37 }}}}/{{{{ val.era5.zarr37 }}}}'
          select: {flow(pl + PRECURSORS)}
        - '{{{{ val.era5.dir37 }}}}/{{{{ val.era5.zarrstl1 }}}}'
      drop: [] # aifs2-1deg drops q_10 (Bo's set); here q_10 is a diagnostic of every run
"""
    open(f"{OUT}/aifs37-{run}.yaml", "w").write(text)
    print(run, len(levels), "levels,", len(SFC) + len(pl) + len(PRECURSORS) + 1, "variables")
