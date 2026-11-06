# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research project comparing next-day NVDA log-return prediction with and without news-derived features. The work is exploratory, so most of the logic lives in notebooks. There is no build step, test suite, linter, or CI.

## Environment

Python 3.12, Windows, with a virtualenv at `.venv`.

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- `requirements.txt` is UTF-16 encoded, so git treats it as binary and shows no diff. Don't rewrite it in another encoding by accident.
- It is incomplete: the notebooks also need `keras`, `tensorflow`, `openai`, and `pandas_market_calendars`, which are installed in `.venv` but not listed.
- Copy `.env.example` to `.env` for API keys. `aquire_data/collect_news_gnews.py` reads `GNEWS_KEY`, while `.env.example` lists `GNEWS_API_KEY`.
- `data/`, `.env`, `*.csv`, `*.txt` (except `requirements.txt`), and `*.json` are gitignored. Datasets exist only locally, and `notes.txt` holds untracked research notes (data-source limits, planned approaches).

## Running things

**Notebooks** are the main entry point. Run them in VS Code with the `.venv` kernel and the notebook's own folder as the working directory, because they use relative paths such as `../../../data/...`. Cells depend on earlier state, so run them top to bottom.

**Data collection scripts** in `aquire_data/` write to paths relative to the working directory, so where you run them decides where the output lands:

```powershell
cd data
python ..\aquire_data\alpaca_news.py               # writes data_alpacanews\
python ..\aquire_data\get_all_news_stocknewsapi.py # writes data_stocknewsapi\
python ..\aquire_data\the_news_api.py              # writes data_news_api\

cd data\data_gnews_api
python ..\..\aquire_data\collect_news_gnews.py     # writes <company>_news.csv, plus progress and .quota.json files
```

Stooq price data is not fetched by any script. It is read from `data/data_stooq/daily/us/nasdaq stocks/2/nvda.us.txt`.

Howeve, dont fetch any new data unless explicitly stated as it should be inside data folder

## Architecture

### Experiment tracks

Each folder under `training/` is one track, kept comparable to the others. The notebooks refer to them as "track 1/2/3":

| Track | Inputs |
| --- | --- |
| `no_news_no_feature_engineering` | Log return and cyclical date encodings (the baseline) |
| `no_news_with_feature_engineering` | Baseline plus technical indicators (`CLOSE_TO_TEMA_10`, `EMA_5_20_SPREAD`) |
| `news_with_feature_engineering` | Baseline plus LLM-derived rolling news features |

Two more approaches are planned in `README.md` but don't exist yet: `news_features_from_rag` and `news_shaped_based_rag`. The agreed order of work is in `ROADMAP.md`.

The real code is in each track's `notebooks/training_exploration.ipynb`, and shared logic such as `make_windows` is copied between notebooks rather than imported. These files are empty placeholders: every track's `model.py` and `training/preprocessing/__init__.py`.

`training/benchmarking/` holds a shared pipeline for comparing variants: `data.py` builds each track's features and windows, `models.py` defines the models, and `experiment.py` has `run_experiment(ExperimentConfig(...))`, which only scores the test split with `evaluate_test=True`. `ExperimentConfig` can also limit every track to the news track's dates (`common_dates=True`) and standardize inputs and target using training rows only (`standardize=True`). Both are off by default to match the notebooks. Runs are compared in `notebooks/model_comparison.ipynb`, with results saved to `results/`.

`walk_forward.py` is the evaluation to use for decisions: `run_walk_forward(WalkForwardConfig(...))` retrains over 8 rounds, never loads the locked test split, and can resume from its results file. Scored windows can use earlier days as inputs, but no training target reaches a scored day. The 80/10/10 split stays for `check_notebook_match` and the final test. After changing it, run:

```powershell
python -m training.benchmarking.check_walk_forward
```

After changing the pipeline's data or default model behaviour, confirm it still reproduces the notebooks (no API calls or cost):

```powershell
python -m training.benchmarking.check_notebook_match
```

### Conventions shared by the notebooks

Keep these consistent across tracks, or results stop being comparable:

- **Target:** `LOG_CLOSE_RETURN`, summed over `horizon_days` after the context window (default 1).
- **Context window:** 30 days (`WINDOW_DAYS`).
- **Split:** chronological 80/10/10 with no shuffling. Windows are built separately within each split.
- **Scaling:** only the news track scales anything: `NEWS_ARTICLE_COUNT_5D`, with the scaler fit on train only.
- **Models:** LSTM (Keras) and XGBoost on flattened windows, compared against `DummyRegressor(strategy="mean")`.
- **Metrics:** MSE, MAE, prediction/actual correlation, and directional accuracy. Judge directional accuracy against the "always up" base rate, not 50%.

### News feature pipeline

This flows through `news_with_feature_engineering`, using GNews data (`data/data_gnews_api/nvidia_news.csv`):

1. **Trading-day alignment (notebook):** timestamps are converted to US/Eastern. Articles published at or after 16:00 roll to the next day, then to the next NASDAQ session (`pandas_market_calendars`), giving `trading_day`.
2. **LLM scoring (`utils/llm_news_features.py`):** one OpenAI Structured Outputs call per `trading_day`, with all of that day's articles in one prompt and returning sentiment, event type, materiality, and similar fields.
3. **Rolling aggregation (`utils/rolling_news_features.py`):** features are reindexed to calendar days, no-news days get neutral values, and 5-day aggregates become `NEWS_*_5D` columns. They are then shifted by one day so day *t* only sees news through *t-1*.
4. **Merge:** an inner join onto the stock frame on `DATE`, done before the split.

The one-day shift and the after-close roll are what prevent look-ahead leakage, so preserve them when changing this pipeline.

### LLM cache

LLM calls cost real money, so results are cached in `news_with_feature_engineering/notebooks/news_features_cache.jsonl`, which is committed to git. `run_experiment` only reads this cache and never calls the API.

- The cache key is the date alone. Changing `MODEL`, `SYSTEM_PROMPT`, the schema, or the ticker does not invalidate it, so stale rows get reused unless you move or delete the file. The prompt is also hardcoded to NVDA.
- `extract_daily_news_features` returns everything in the cache, not just the dates in the input frame.
- `OpenAI()` is constructed before the cache is checked, so `OPENAI_API_KEY` must be set even on a fully cached run. The notebook doesn't call `load_dotenv`, so the key has to already be in the environment.
