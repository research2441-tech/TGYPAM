# TGYPAM Crop Growth Yield Profit Reproducibility Implementation

This repository is an **independent executable operationalization** of the Time-Centric Growth–Yield–Profit Analysis Model (TGYPAM) described in the manuscript *Big Data Driven Deep Learning for Integrated Crop Growth Yield Profitability Prediction and Resource Optimization in Smart Agriculture*. It supports research audit, implementation transparency, and follow-up evaluation.

## Scope and limitations

TGYPAM combines Time-Centric Growth Analysis (TCGA), Time-Centric Yield Analysis (TCYA), Time-Centric Profit Analysis (TCPA), Plant Selection Score (PSS), Fluid Factor (FF), and Fertilizer Requirement Factor (FRF). The manuscript describes the factor classes but does **not** provide a fully reproducible layer-wise network definition or sufficiently specified evaluation metrics. Consequently this code explicitly labels its GRU multi-task architecture and normalized support aggregation as **operationalization choices**, not a verified reconstruction of unknown original implementation decisions. Numeric coefficients and hyperparameters in `config.yaml` are reproducible defaults, not necessarily those used by the study authors. Some embedded equations in the source manuscript are not available as unambiguous executable formulas; see the distinctions below.

The manuscript reports scores 99, 97, 98 at one million records, but the original labeled dataset, exact data dictionary, experiment scripts and definitions of these percentages are not provided. The values in `reported_results.csv` are transcriptions of manuscript tables, not newly measured output. No code can independently verify those scores from the provided documents alone. **Do not describe a synthetic run or a run with another dataset as exact reproduction of the manuscript.** Baseline entries are reference scores only; IMCDMM, MLACPP and CERES baseline models are not implemented.

## Architecture

All source files occupy the repository root: `config.yaml`, `pipeline.py`, `tgypam.py`, `train.py`, `evaluate.py`, `reproduce.py`, `generate_demo_data.py`, `tests.py`, `reported_results.csv`, `requirements.txt`, and `CITATION.cff`.

- `pipeline.py` validates schema, checks duplicates, splits by global chronological timestamps, fits imputation/scaling on training only, and creates within-split fixed-length time windows per field.
- `tgypam.py` implements normalized support-factor aggregation; the `TCGA` and `TCYA` numerical factor functions accept soil, fluid, environment and climate support inputs as [0,1] quantities. It implements `TCPA` as revenue minus production cost (USD/ha). `PSS` uses a weighted geometric product of normalized growth, yield and net-profit supports.
- `TGYPAM` is a fully specified GRU sequence encoder with shared representation, crop embedding, and growth/yield/profit regression heads. These are practical design selections for testing the proposed modular integration, not established original network details.
- FF and FRF are **historical water and fertilizer application averages** (mm and kg/ha), never direct agronomic prescriptions.
- `evaluate.py` reports MAE, RMSE, R², and exploratory PSS summaries. PSS cannot be interpreted as validated crop-counterfactual suitability without outcomes for alternative crops under the same conditions.

## Getting started

Python 3.10+ recommended.

```bash
python -m pip install -r requirements.txt
python generate_demo_data.py --fields 40 --months 60 --out agriculture.csv
pytest -q tests.py
python train.py --config config.yaml
python evaluate.py --config config.yaml
```

The generated CSV is **synthetic demonstration data**, and its metrics have no evidential bearing on the manuscript. Output files created by the above commands are local artifacts and are not included in the repository ZIP.

For the real dataset, replace `agriculture.csv` with an ethically shareable genuine dataset and edit all feature/target names, units and time/field keys in `config.yaml`. The default fields include: timestamp, field_id, crop, fourteen numeric features (`temperature_c`, `humidity_pct`, `rainfall_mm`, `soil_ph`, `soil_n`, `soil_p`, `soil_k`, `irrigation_mm`, `soil_moisture_pct`, `sunlight_hours`, `wind_kmh`, `field_area_ha`, `market_price_usd_t`, `input_cost_usd_ha`), three supervised labels (`growth`, `yield_t_ha`, `profit_usd_ha`), and two resource observations (`water_mm`, `fertilizer_kg_ha`). This is an **illustrative schema**, not an assertion that the manuscript's undisclosed 20-feature schema has these exact names.

**Forecasting assumption:** each row uses only features known at its timestamp. For genuine future-horizon forecasting, shift labels forward *within field* by the target horizon, remove look-ahead features, and revalidate all partition boundaries. The default is a same-time supervised temporal assessment, not independently validated next-season yield forecasting. A crop identifier remains constant for each field in the synthetic generator. Source data should document plantings that change crops and partition sequence construction accordingly.

## Size-specific runs

```bash
python reproduce.py --config config.yaml
```

Runs at 300,000 / 500,000 / 1,000,000 **source rows only when available**; otherwise it skips sizes. The prefix sampling is chronological and should not be mistaken for the original study's undisclosed sampling protocol. Because this subsampling can truncate the later test horizon or crop representation, some sizes may not permit meaningful evaluation.

## Measurement and disclosure checklist

1. Record the exact training dataset provenance, collection period, geographical coverage, crop classes, class counts, and data dictionary.
2. Define the numerical target for growth and its units; separate growth regression from classification/selection success.
3. Explain the experimental meaning and formula for each manuscript performance percentage before treating it as a directly comparable measured result.
4. Report identical baselines on identical leakage-resistant splits, training budgets, hyperparameter tuning and performance metrics. Manuscript reference baselines alone do not satisfy this.
5. Obtain cross-region and cross-season external validation, field-level clustered confidence intervals, and agronomic verification of FF and FRF.
6. Verify all formulas against the actual source equation objects and the authors' training records before claiming exact source-code equivalence.

## Code availability

A GitHub source archive alone is not a DOI-assigning repository. If the journal/editor requires archival deposition, publish a specific GitHub release, link that release to an authorized Zenodo archival record, and add the resulting **actual assigned** persistent identifier and commit hash in the manuscript Code Availability statement. This package deliberately does not invent a DOI.
