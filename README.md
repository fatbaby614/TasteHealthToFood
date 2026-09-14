# TasteHealthToFood

Analysis code and result artifacts accompanying the manuscript:

> **A reproducible and auditable pipeline for single-trial EEG decoding using Riemannian tangent-space regression and multi-seed permutation validation**
>
> Huang Tan, Yajing Liu, Weiqian Hu, Xiao Liang, Guangqiang Yin (corresponding author: yingq@uestc.edu.cn)
>
> Submitted to *Computer Methods and Programs in Biomedicine*.

## What this repository contains

| Path | Contents |
|---|---|
| `code/` | The complete analysis pipeline: behavioral modeling (logistic + DDM/HSSM), EEG preprocessing, ERP and time-frequency analyses, Riemannian tangent-space decoding, sliding-window decoding, SVR window-by-target decoding with permutation testing, cross-task analyses, and permutation-stability audits. |
| `code/run_all.sh` | One-command replication script that reruns every stage with the exact parameters reported in the paper. |
| `outputs/results/` | Every JSON/CSV result artifact cited in the manuscript, plus per-window intermediate files of the multi-seed permutation stage. |
| `outputs/figures/` | Manuscript figures regenerated from the result artifacts. |
| `outputs/qc/` | EEG quality-control summaries. |
| `paper/` | LaTeX sources of the manuscript (Elsevier CAS `cas-dc` template). |

Subject-level data are **not** included. Raw EEG must be obtained from OpenNeuro
(below); preprocessed epoch caches (`data/epochs/*.npy|*.npz`) and run logs
(`outputs/logs/`) are generated locally and excluded from version control.

## Dataset

The pipeline analyzes the **FoodEEG** dataset:

- OpenNeuro accession `ds007012`, version 1.1.0 used in this study
- https://openneuro.org/datasets/ds007012/versions/1.1.0
- 64-channel EEG (BioSemi ActiveTwo, 512 Hz) from 117 participants performing a
  food categorization task and a binary food-choice task; on choice trials
  participants rated each food item for tastiness, healthiness, and willingness
  to eat (0–100 scales).

Download the dataset from OpenNeuro and point the pipeline at its root:

```bash
export DATA_DIR=/path/to/foodeeg-ds007012
```

`code/config.py` reads `DATA_DIR` and `code/eeg_preprocessing.py` reads
`FOODEEG_DATASET_DIR`; `run_all.sh` sets both from `DATA_DIR` automatically.

## Environment

Python 3.12+ (developed on 3.13; `setuptools>=70` recommended), then:

```bash
pip install -r requirements.txt
```

Key dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `mne`,
`scikit-learn`, `pyriemann`, `matplotlib`, and `hssm`/`pymc`/`arviz` for DDM
sampling. The pinned combination `hssm==0.3.0, pymc==6.0.1, arviz==1.2.0`
matches the compatibility shims in `code/ddm_model.py`; if you upgrade any of
the three, validate them together.

## Quick start

From the repository root (the replication script requires a bash environment;
the individual Python stage scripts can also be run manually on any OS):

```bash
chmod +x code/run_all.sh

bash code/run_all.sh                # full replication with the paper's parameters
bash code/run_all.sh --quick        # smoke test (few subjects, reduced permutations)
bash code/run_all.sh --only 1,5,6   # run selected stages only (numbers or names)
bash code/run_all.sh --skip 7       # skip a stage
bash code/run_all.sh --force-preproc  # ignore epoch cache, reprocess from raw data
bash code/run_all.sh -h             # full usage
```

Paper-critical permutation parameters are built in as defaults:
100 permutations × 5 seeds, per-subject training budget of 200 trials
(the values recorded in `outputs/results/svr_roi_channels_summary.json`).

### Pipeline stages

| # | Name | What it runs |
|---|---|---|
| 0 | `envcheck` | Checks Python packages/versions, dataset path, and epoch cache |
| 1 | `behavior` | `load_behavior` → `compute_scsr` → `baseline_model` → `ddm_model` |
| 2 | `preproc` | EEG preprocessing (cached per subject) → `attach_ratings` → `eeg_qc_report` |
| 3 | `univariate` | ERP (`erp_analysis`) and time-frequency (`timefreq_analysis`) analyses |
| 4 | `integrate` | Neural–behavior integration (`sullivan_integration`) |
| 5 | `decoding` | Riemannian classification, sliding-window decoding, SVR decoding, and the 18-cell window×target permutation family (six windows in parallel, merged with whole-family FDR) |
| 6 | `extended` | ROI-channel analysis and cross-task analyses (the paper's core results) |
| 7 | `audit` | Permutation-test stability audit (`verify_temporal_perm`) |
| 8 | `check` | Verifies that all result artifacts expected by the manuscript are present |

A full run takes several hours on a workstation (DDM sampling and the
multi-seed permutation stages dominate). Running overnight inside `tmux` is
recommended:

```bash
tmux new -s rerun './code/run_all.sh'
```

Per-run logs, including a step-by-step `summary.txt` with exit status and wall
time, are written to `outputs/logs/run_all_<timestamp>/`.

## Verifying a replication run

After a full run, compare the reported statistics against the manuscript via
`outputs/results/`, in particular `svr_roi_channels_summary.json`,
`svr_summary.json`, `riemannian_summary.json`, `ddm_summary.json`, and
`svr_scsr_groups_summary.json`. Stage 8 already asserts that every expected
artifact exists; the audit stage (7) reproduces the seed-stability analysis of
permutation p-values reported in the paper.

## Citation

If you use this code or the results reported here, please cite the
accompanying manuscript:

> Tan H., Liu Y., Hu W., Liang X., Yin G. *A reproducible and auditable
> pipeline for single-trial EEG decoding using Riemannian tangent-space
> regression and multi-seed permutation validation.* Submitted to Computer
> Methods and Programs in Biomedicine.

## License

The analysis code in this repository is licensed under the MIT License
(see [LICENSE](LICENSE)). The FoodEEG dataset is not redistributed here and
remains subject to the terms under which it is shared on OpenNeuro.

## Notes

- Comments in some scripts are written in Chinese; `bash code/run_all.sh -h`
  prints the full usage help in Chinese.
- Questions: open an issue, or contact the corresponding author
  (yingq@uestc.edu.cn).
