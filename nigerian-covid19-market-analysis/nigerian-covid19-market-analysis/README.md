# The Impact of COVID-19 on Nigerian Financial Markets

**A Statistical Analysis of Market Regime Shifts Using a Hidden Markov Model (2017–2025)**

MScFE 690 Capstone Project, WorldQuant University. Practical track: Statistical Characteristics of Markets.

## Project summary

The project studies how Nigerian equity (NGX All-Share Index), foreign-exchange (USD/NGN), sovereign bond (FGN 2Y, 5Y, 10Y and 20Y yields) and oil (Brent crude) markets behaved before, during and after COVID-19, and tests whether a Hidden Markov Model (HMM) fitted to NGX returns and volatility can provide a regime indicator that works out of sample.

Research questions:

1. **RQ1.** Did returns, volatility, distribution shape, cross-market correlation and yield-curve structure change significantly between the Pre-COVID, COVID and Post-COVID periods?
2. **RQ2.** Does an HMM identify the COVID window as a separate high-volatility regime, and do markets afterwards return to their Pre-COVID regime?

Main results (full discussion in the project report):

* Variance differs significantly between periods in 18 of 24 comparisons; mean returns differ in only 5.
* The share of NGX days in the high-volatility state falls from 49.87% (Pre-COVID) to 32.60% (COVID) and 23.74% (Post-COVID), so COVID-19 is not identified as a distinct high-volatility regime for Nigerian equities. The decline is steepest after the June 2023 naira float (32.96% to 18.33%).
* A three-state HMM fits better than a two-state HMM (BIC 8,022 against 9,438) and gives the same direction.
* Out of sample (parameters frozen before the test window), the filtered regime probabilities agree with the full-sample model on 97–99% of test days and a high-volatility flag precedes higher realised volatility. A simple volatility threshold anticipates volatility about as well but switches regime more often.
* Cross-market return correlations are small (largest 0.078, NGX–Brent).

## Repository structure

```
nigerian-covid19-market-analysis/
├── README.md                  This file
├── requirements.txt           Python dependencies (pip install -r requirements.txt)
├── config.json                Analysis settings (dates, periods, windows, seeds)
├── CONTRIBUTORS.md            Team members and roles
├── .gitignore
├── data/
│   └── raw/                   The four input CSV files (see "Data" below)
├── notebooks/
│   └── Nigeria_COVID_Final_Pipeline.ipynb   Full pipeline, with the outputs of the run used in the report
├── src/
│   └── nigeria_covid_pipeline.py            The same pipeline as a script
└── results/                   Created on each run: tables (CSV) and figures (PNG)
```

## Requirements

* Python 3.10 or newer
* The packages in `requirements.txt` (numpy, pandas, scipy, matplotlib, scikit-learn, statsmodels, arch, hmmlearn, jupyter)

## How to run

### Option A: locally (instructor or reviewer)

```bash
git clone <HTTPS link of this repository>
cd nigerian-covid19-market-analysis
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/nigeria_covid_pipeline.py
```

The script reads the CSV files from `data/raw/`, prints every table to the console, and writes all tables, figures and a `project_results.zip` to `results/`. It takes a few minutes, mainly for the repeated HMM fits.

To run the notebook instead, start Jupyter from the repository root (`jupyter notebook`), open `notebooks/Nigeria_COVID_Final_Pipeline.ipynb` and choose Run All Cells. The notebook already contains the outputs of the run reported in the paper.

### Option B: Google Colab

1. Open `notebooks/Nigeria_COVID_Final_Pipeline.ipynb` in Colab.
2. Upload the four CSV files from `data/raw/` to `/content/`.
3. Choose Runtime, then Run all. The notebook installs `arch` and `hmmlearn` itself and downloads `project_results.zip` at the end.

## Configuration

`config.json` holds the settings used by the pipeline:

| Key | Meaning | Value |
|---|---|---|
| `start_date`, `end_date` | Analysis window (the Brent file extends into 2026 and is cut at `end_date`) | 2017-01-01, 2025-12-31 |
| `periods` | Pre-COVID, COVID and Post-COVID boundaries | see file |
| `naira_float_date` | Date used to split the Post-COVID period (CBN FX-window unification) | 2023-06-14 |
| `n_restarts` | Random restarts for the HMM model comparison and the out-of-sample fits | 10 |
| `fwd_window` | Forward realised-volatility horizon for the out-of-sample test (trading days) | 20 |
| `rolling_window` | Rolling volatility window (trading days) | 20 |
| `hmm_random_state` | Seed for the headline two-state HMM | 42 |
| `bond_columns` | Names of the four bond-yield columns | see file |

If `config.json` is missing (for example in Colab), the notebook uses the same values as built-in defaults.

## Pipeline steps

| Step | Content |
|---|---|
| 1–2 | Load the four CSV files; clean and cap to the analysis window; drop missing prices before computing returns; clean each bond tenor separately and build the jointly complete bond sample |
| 3–4 | Descriptive statistics, daily log returns, 20-day rolling volatility |
| 5 | ARIMA(1,0,1) and GARCH(1,1) on NGX returns |
| 6 | PCA on standardised daily yield changes |
| 7 | Correlations on returns and yield changes |
| 8 | Two-state Gaussian HMM; states labelled by mean realised volatility |
| 9–10 | Per-tenor bond cleaning; Welch's t-test and Levene's test for all 24 series and period pairs |
| 11 | Split of the Post-COVID period at the 2023 naira float |
| 12 | Two-state versus three-state HMM (AIC, BIC) |
| 13 | Stationarity tests (ADF and KPSS) |
| 14 | Correlations by period with confidence intervals and change tests |
| 15 | Out-of-sample HMM (frozen parameters, filtered probabilities, forward test) |
| 16 | Economic significance (VaR), threshold benchmark, sensitivity to window length and initialisation, alternative float date |
| 17 | The 13 figures used in the report (`results/figures/`) |
| 18 | Zip of all results |

## Outputs

Everything is written to `results/`, including `welch_levene_full_table.csv`, `correlation_by_period.csv`, `correlation_change_tests.csv`, `hmm_2state_regime_shares.csv`, `hmm_3state_regime_shares.csv`, `hmm_2_vs_3_state_fit.csv`, `hmm_oos_summary.csv`, `economic_significance_var.csv`, `benchmark_threshold_vs_hmm.csv`, `hmm_sensitivity.csv`, `stationarity_adf_kpss.csv` and `figures/fig01_series.png` to `fig13_brent.png`.

## Data

| File in `data/raw/` | Series | Source |
|---|---|---|
| `NGX All-Share Index (ASI).csv` | NGX All-Share Index, daily | Nigerian Exchange (NGX) |
| `USD_NGN Historical Data.csv` | USD/NGN, daily | Central Bank of Nigeria |
| `Nigeria_Bond_Yields_2017_2025_Redone.csv` | FGN yields 2Y, 5Y, 10Y, 20Y, daily | CBN / FMDQ |
| `Brent Oil.csv` | Brent crude, daily | FRED series DCOILBRENTEU |

Cleaned sample sizes: NGX 2,208; USD/NGN 2,326; Brent 2,283; bond dates with all four tenors 1,319 of 2,198.

## Reproducibility notes

* The headline HMM uses `random_state = 42`; model comparisons use the best of 10 random restarts. Small differences in the last decimals can occur between `hmmlearn` versions.
* hmmlearn's state numbers are arbitrary, so states are always relabelled by mean realised volatility before any result is reported.
* Overlapping 20-day windows and serially dependent regime labels make p-values for rolling volatility and regime shares approximate; this is stated in the report.

## Git workflow

* Commit messages use the imperative mood with a short summary line (for example `Add out-of-sample HMM validation`), followed by a body when the reason is not obvious.
* Each commit contains one logical change; large data or result files are not committed except the four input CSVs.
* Group members push to feature branches (`feature/<topic>`) and merge into `main` through pull requests.


## Authors

See [CONTRIBUTORS.md](CONTRIBUTORS.md).
