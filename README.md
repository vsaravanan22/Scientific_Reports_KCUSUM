# KCUSUM experiments for the Scientific Reports manuscript

Code for every experiment in "A systematic evaluation of kernel cumulative sum
change detection for real-time molecular dynamics streams": the controlled
benchmarks, the 1H9T rigid-body diagnostics, the KCUSUM / BOCPD / KLIEP
comparison, and the 1FME centre-of-geometry check.

```
pip install -r requirements.txt
```

## Files

| File | Purpose | Manuscript |
|---|---|---|
| `kcusum_core.py` | KCUSUM in R^d with a Gaussian kernel (Numba), the four change tasks, MMD^2 estimation, bounds (eqs 8-9) | Methods: KCUSUM, bounds |
| `run_benchmark.py` | Time to false alarm, delay, log fits, Figure 1 | Table 2, Fig. 1 |
| `md_io.py` | Streaming VMD XYZ reader, stops at an incomplete final frame | Methods: MD data |
| `rigid_body.py` | Kabsch superposition, unaligned vs superposed step change, RMSD from frame 1, wrapping check | Fig. 2 |
| `kcusum_md.py` | Detector loop of the public `kcusum.py`, `fitter` reference, 200 seeded runs | Table 3 (KCUSUM) |
| `bocpd.py` | BOCPD, normal-inverse-gamma prior, hazard 1/50, MAP run-length alarm rule | Table 3 (BOCPD) |
| `kliep.py` | Online adjacent-window KLIEP, 95th-percentile calibrated threshold | Table 3 (KLIEP) |
| `run_1h9t.py` | Full 1H9T analysis | Figs 2-3, Table 3 |
| `run_1fme.py` | Centre-of-geometry steps around the earlier 1FME candidates | Results: 1FME |
| `make_test_trajectory.py` | Synthetic XYZ file for testing the pipeline without the real data | - |

## Running

Controlled benchmarks (5,000 runs per cell, 4e6 pairs for MMD^2):

```
python run_benchmark.py --outdir results/benchmark
python run_benchmark.py --quick      # 200 runs, h <= 16, for a quick check
```

Numba parallelizes over runs. Every run seeds its own generator from a fixed
seed per (task, threshold), so results do not depend on the number of cores.
Task (4) at h = 56-64 is the most expensive setting.

1H9T (needs the trajectory from BNL):

```
python run_1h9t.py data/1H9T.xyz --runs 200 --outdir results/1h9t
```

1FME:

```
python run_1fme.py data/1FME.xyz --offset 0 --outdir results/1fme
```

`--offset` maps the candidate frame numbers of the earlier 26,283-frame analysis
onto this file. Set it once it is confirmed which frames that analysis used.

Pipeline test without real data:

```
python make_test_trajectory.py test.xyz
python run_1h9t.py test.xyz --runs 20 --outdir results/test
```

## Conventions

* Frames are numbered from 1. The frame-to-frame RMSD at frame t compares frames
  t-1 and t, so that signal starts at frame 2.
* Calibration (reference) period is frames 1-80 for all three detectors.
* KCUSUM on 1H9T uses h = 0.015, delta = 0.1 and sigma^2 = 1 on unstandardized
  signals in Angstrom, as in the public repository. Alarms fall on odd 1-based
  frames because observations are paired.
* Parsed trajectories are cached as `<file>.npy` next to the input.
