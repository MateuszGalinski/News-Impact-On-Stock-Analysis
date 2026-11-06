# Roadmap

Agreed plan as of 2026-11-06. Steps 1, 2 and 3a are built and have been run. Step 3c is in progress. Steps 3b, 4 and 5 haven't started.

**Project aim:** show how unstructured data (news) can be combined with structured data (prices), and how well RAG suits that task.

**Starting point:** Phase 2 in `training/benchmarking/notebooks/model_comparison.ipynb`, which uses common dates and standardization. It fixed training, but no model has a dependable edge over the baseline on validation. Phases 1 and 2 stay in the notebook as the record.

**Rules for every step:**

- Keep the test split locked until the final comparison, and use it once.
- Fix the options to compare before seeing results, and report all of them.
- Judge directional accuracy against always-up, and MSE and MAE against the train-mean baseline.
- Preserve the one-day news shift and the after-close roll, which prevent look-ahead leakage.

## 1. Walk-forward evaluation

Every model is currently judged on the same 123 days in 2025, which is too few to separate real differences from noise. This step comes first because it decides how every later change gets judged. It has no API cost.

- Train on an early period, score the next stretch, then extend the training period and repeat.
- Refit the scaling each round, on that round's training data only.
- Use only the first 90% of the data, so the test split stays locked. That's about 1,380 days. Starting with about 600 for training gives about 780 scored days, in roughly 8 rounds.
- Report confidence intervals that account for returns on nearby days being related.
- Save results as the run goes. Each model trains once per round for each seed, so runs will be slower. Use fewer seeds if it takes too long.

First run: put the Phase 2 setup through walk-forward, before changing any models, to see how much of the current results survive.

**Result (2026-11-06):** none of the 9 setups beat the baseline. Over 781 scored days (2022-05-25 to 2025-07-08), no seed of any setup beat the train-mean baseline on MSE, and every setup averaged below always-up (54.9%) on direction. The best MSE result, XGBoost v2 on news, has an out-of-sample R² of -0.027 (95% interval -0.047 to -0.010). The small positive results on the old 123 validation days didn't survive. The confidence intervals are about ±4 points on direction.

## 2. Fair models

No API cost.

- Use one standardized LSTM and one XGBoost, with the same setup on every track. From here we compare inputs, not architectures.
- Retire XGBoost v2's separate settings. Drop the RNN from the comparison unless walk-forward gives a reason to keep it.
- Run early stopping on its own slice from the end of each training period, not on the data being scored. This removes the optimism in XGBoost v2's current result.

**What was built:** results are in `training/benchmarking/notebooks/unified_models.ipynb`.

- One XGBoost setup (v2's settings, full 30-day windows), and early stopping for RNN, LSTM and XGBoost. The RNN was kept.
- A fourth input set, `news_with_feature_engineering_and_tech`, because the existing news track has no technical indicators. News vs no news, no features is therefore the clean test of news.
- Both evaluations: walk-forward (781 days) as the stronger evidence, and 80/10/10 (123 validation days) as a second view.
- Direct day-by-day comparisons between input sets, for each model.

**Result (2026-11-06):**

- **Against the baseline:** on walk-forward, no seed of any of the 12 setups beat the train-mean baseline on MSE, and every setup averaged below always-up (54.9%) on direction. The best is news and indicators with XGBoost, with out-of-sample R² of -0.020 (95% interval -0.036 to -0.006).
- **News vs no news:** on walk-forward, all three models' MSE and direction intervals include zero in the main comparison, so news shows no detectable benefit or harm. On 80/10/10, the LSTM and RNN are worse on direction with news, and the LSTM is worse on MSE. XGBoost shows no difference.
- **Early stopping:** it cut the worst MSE losses, but predictions are flatter (spread 0.11-0.31 of actual) and correlation with actual returns is about zero. Many fits barely train: on the news input sets, the LSTM's best epoch is within the first 3 in 75% of walk-forward fits, and is epoch 1 in about 40-50%.

**Learning curves (2026-11-06):** `training/benchmarking/notebooks/learning_curves.ipynb` ran the unified models with early stopping off, on rounds 1, 4 and 8, with 2 seeds.

- **Overfitting, not underfitting:** by the last step, training loss falls to about 0.003-0.10 of the baseline's on average in every round. RNN and LSTM loss on the stopping slice rises well above it, to about 1.9-2.0 and 2.7-6.5 times the baseline.
- **More depth or training isn't the fix:** the models already fit their training data almost fully.
- **Input sets:** the news LSTMs pick about epochs 1-2, compared with about 9-13 for the no-news LSTMs, and they never get below the baseline on the stopping slice on average.
- **XGBoost:** it stays close to the baseline on the stopping slice, about 0.94-1.14 by the last tree, without the sharp rise. Its best points are slightly below 1, but they're chosen on that same slice and didn't carry over to the scored days.

## 3. A stronger baseline

The learning curves show overfitting, so this step controls that first, then tests which news features matter. The winner becomes the control for RAG.

### 3a. Regularization test

`training/benchmarking/notebooks/regularization.ipynb`, using 3 seeds on walk-forward and 80/10/10. The unified models are the control, and every candidate keeps early stopping.

| Candidate | Change | Models |
|---|---|---|
| Dropout 0.2 | Mild dropout in the recurrent layer | RNN, LSTM |
| Dropout 0.5 | Strong dropout in the recurrent layer | RNN, LSTM |
| L2 0.0001 | Mild weight penalty | RNN, LSTM |
| L2 0.001 | Stronger weight penalty | RNN, LSTM |
| Smaller network | 16 recurrent units and Dense(8) | RNN, LSTM |
| 2-layer LSTM | Layers of 64 and 32 units, as a test of depth | LSTM |
| Stronger XGBoost | Depth 2, learning rate 0.02, row and column sampling 0.5, `reg_lambda` 10, `min_child_weight` 10 | XGBoost |

Real progress means correlation clearly above zero, an R² interval reaching or crossing zero, and direction closer to or above always-up. Wider predictions, or better MSE from flatter predictions, aren't enough. A winner needs a 5-seed rerun before it's trusted.

**Result (2026-11-06):** no candidate met the criteria, so there's no winner and no 5-seed rerun. On walk-forward, no seed of any of the 60 setups (including controls) beat the train-mean baseline on MSE, and every setup averaged below always-up (54.9%) on direction.

- **Better scores came from flatter predictions:** across the 60 setups, narrower predictions go with better R² (rank correlation -0.91). 17 of the 48 comparisons with the control improve MSE, but that's the kind of gain the criteria rule out.
- **Stronger XGBoost:** it has the best R², -0.0006 (95% interval -0.0043 to 0.0031) with no news and no features, but it's close to predicting the mean. Its spread is 0.05 of actual, about 92% of its predictions are up, and its correlation is at or below zero. That makes it a weak tool for testing features in 3b. Its patience stopped it early, but the rerun below shows longer patience is worse.
- **Depth:** the 2-layer LSTM didn't help, and was much worse on no news with indicators, at -0.123 (-0.246 to -0.035).
- **A weak lead in the RNNs:** 9 of 60 correlation intervals are above zero, and all 9 are RNNs. The best control is no news with indicators, at 0.087 (0.021 to 0.156). It's positive in all 8 rounds and doesn't depend on the largest moves. Its R² is still -0.047, partly because its predictions are too wide, and it was picked from 60 setups. Shrinking its predictions by the best amount in hindsight gives an R² of about +0.008, which isn't a fair result.
- **News vs no news:** in the 15 comparisons, news was never better on MSE or direction. It was worse on MSE in 7 (5 RNN, 2 LSTM) and on direction in 1. With these 3 seeds, the RNN control's MSE and the XGBoost control's direction are worse with news, but both intervals included zero with 5 seeds, so individual results depend on the seeds.
- **80/10/10:** 19 of the 20 best setups have no news. The best is the LSTM with dropout 0.2 on no news with indicators, with R² 0.022 and direction at 54.5% against 55.3% for always-up.

**Learning curves (2026-11-06):** `training/benchmarking/notebooks/regularization_curves.ipynb` ran every candidate with early stopping off, on rounds 1, 4 and 8, with 1 seed. Some candidates clearly reduce overfitting, but none shows clear learning on unseen days.

- **Overfitting:** by the last epoch, dropout 0.5 cuts stopping-slice loss from 1.92 to 1.07 times the baseline for the RNN, and from 4.12 to 1.09 for the LSTM. Dropout 0.2 and the smaller network help less. The L2 settings tested do almost nothing, and the 2-layer LSTM still reaches 3.39.
- **Learning:** at early stopping's pick, every candidate averages 0.98-1.05 on the stopping slice. Being below 1 there doesn't carry over: the XGBoost control is below 1 in all 12 fits, but its walk-forward R² is about -0.03.
- **Stronger XGBoost:** a patience of 30 trees is too short for a learning rate of 0.02. It picks about 29 trees on average, while in rounds 4 and 8 the loss reaches its lowest after about 400-470 trees. Replaying the curves with a patience of 75 gives about 235 trees, and a stopping-slice loss of 0.974 instead of 0.996. Round 1 gets worse with more trees. The scored-day test is below.

**Stronger XGBoost with a patience of 75 (2026-11-06):** `training/benchmarking/notebooks/regularization_patience.ipynb` reran that candidate with one change, using the same seeds. Longer patience made it worse, so short patience wasn't hiding signal, and step 3a still has no winner.

- **Walk-forward:** R² is worse than with a patience of 30 on all 4 input sets, at -0.0075 to -0.0149, and no interval reaches zero. Correlation is -0.030 to -0.042, and every interval includes zero. Direction is 52.4-53.2%, below always-up (54.9%).
- **Stopping-slice gains didn't carry over:** rounds 4 and 8 trained for the most extra trees, but got worse on scored days. Round 4's R² goes from 0.0002 to -0.027, and round 8's from -0.019 to -0.050.
- **80/10/10:** both settings pick the same trees in all 12 fits, so this split can't separate them.

### 3b. Feature composition test

A separate test, run after 3a on the best stable setup, using the cached news features with no LLM reruns. It replaces testing one hand-picked subset of features, which could give a false positive or a false negative and wrongly close the question.

| Part | What it tests |
|---|---|
| Each feature group alone | Whether a group carries anything by itself |
| All news minus one group | Whether removing a group changes anything |
| Shuffled news (placebo) | What chance looks like. Real news must clearly beat it before any feature finding is believed |

- **Groups, fixed before seeing results:** sentiment (mean, confidence, disagreement), event importance (materiality, surprise, guidance change), and news volume (article count).
- **Also consider:** the LLM's event type and time horizon fields, which the models don't use yet. Check that they're in the cache first.
- **Last-day news:** test it here too, as an alternative to the full window.
- **A negative result** applies only to these features, target and models. It doesn't rule out better news features, such as ones from RAG.

### 3c. Move-size and horizon tests

Next-day returns haven't beaten the baseline in any test so far. Move size, the next day's absolute log return, gives a target with a known pattern, because big moves tend to follow big moves. It answers whether news adds anything to that pattern. The return target stays as it is, and this has no API cost.

The plan below was fixed before any results.

**Stage 1, 1-day move size:** `training/benchmarking/notebooks/move_size.ipynb`, using 3 seeds on walk-forward and 80/10/10.

- **Inputs:** the 4 input sets, each with the day's absolute return added.
- **Baselines:** the train mean, yesterday's move, the 5-day and 20-day average moves, and HAR (Corsi, 2009), a linear regression on the 1, 5 and 22-day average moves. HAR is the main baseline.
- **Models:** RNN and LSTM, as the control and with dropout 0.5. XGBoost, as the control and Stronger with a patience of 30. HAR plus the last day's news features.
- **Measures:** MSE and MAE, Pearson and rank correlation, and how often the biggest 20% of moves are caught, with the cutoff set from each round's known days.

The questions, in order:

1. **Is there a pattern?** HAR's R² interval must be above zero against the train mean, or the test stops there.
2. **Does a trained model add anything?** Its MSE improvement over HAR must have an interval above zero, with MAE and rank correlation not clearly worse.
3. **Does news help?** With the same model, news must beat no news on the same terms. HAR + news vs HAR is the simplest version of this test.

If news wins, run a shuffled-news placebo and the news audit (5a) before trusting it. Article count could just reflect that big days attract more news, and the LLM may know how past events turned out.

**Stage 1 result (2026-11-06):** under the rules above, the test stops at question 1, and news shows no benefit.

- **Question 1:** HAR's R² is 0.012 (95% interval -0.020 to 0.051), so it doesn't pass. Its rank correlation is 0.133 (0.048 to 0.189), so there's a real but small pattern in which days are bigger. Unlike the return target, 13 of the 26 fitted setups, including HAR, have an R² above zero, but none has an interval above zero.
- **Question 2:** none of the 25 comparisons with HAR has an MSE interval above zero, and 6 are clearly worse. The 20-day average ranks days about as well as HAR, at 0.144.
- **Question 3:** in 13 comparisons, news is never better on MSE or rank correlation, and is worse on MSE in 5, including HAR + news vs HAR. It's better on MAE in 2 and worse in 2.
- **Big moves:** they aren't predicted. HAR predicted 0.022 for the largest move, 0.218, and no trained model catches more than about 6% of big moves.
- **Spikes:** checked after seeing results. The biggest 1% of days make up 33% of the train mean's squared error, but removing them only raises HAR's R² to 0.024, so they aren't hiding a stronger pattern.
- **80/10/10:** R² reaches 0.072 and rank correlation 0.36 over 123 days, but news is again never better.

These news features end two days before the scored move's close, and don't include scheduled events such as earnings dates, so this doesn't rule out better news inputs.

**Stage 2, 5-day horizons:** review stage 1 first. Then run 5-day move size, as the average daily move over the 5 days, and 5-day return, each on 749 scored days. Those days overlap, so they're worth only about 150 independent periods. A 20-day horizon was left out, because about 31 independent periods is too few.

## 4. Pretraining on other companies

Fit an LSTM on other companies, then fine-tune it to predict NVDA. Start with no news and with features, which is free. Drop the no-features track for this test. Test news only if pretraining helps.

Compare on the same NVDA walk-forward days:

| Version | Trains on | What it tells you |
|---|---|---|
| A. NVDA only, common dates | NVDA from 2020 | The main control |
| B. NVDA only, full history | NVDA from 1999 | Whether more NVDA data is enough on its own |
| C. Pretrained, no fine-tuning | Other companies only | Whether any general pattern carries over |
| D. Pretrained and fine-tuned | Other companies, then NVDA | The idea being tested |

**Decision rule:** pretraining earns its place only if D beats both A and B, with confidence intervals that support it.

Rules to keep it fair:

- **Time cutoff:** in each round, other companies' data must end before NVDA's scored period starts.
- **Separate scaling:** standardize each company with its own pre-cutoff statistics.
- **Companies fixed in advance:** semiconductors (AMD, AVGO, INTC, MU, QCOM, TSM), and maybe big tech as a second group. These all survived and mostly did well, so mention that bias in the write-up.
- **One fine-tuning method:** a lower learning rate and early stopping on a separate slice, chosen before seeing results.

If D wins, add news in this order:

1. Pretrain the price LSTM on other companies, with prices only. Combine its output with NVDA's existing news features in a small network trained on NVDA. Compare D against D plus news. This needs no new API calls and is close to the "autoencoder plus LLM output" idea in `notes.txt`.
2. Only if that shows promise, pretrain on other companies' news.

Other companies' news needs code changes and API spend. LLM scores exist only for NVDA, the prompt is hardcoded to NVDA, and the cache is keyed by date alone, so it would wrongly reuse NVDA's scores. Raw news exists for Apple, Google, Microsoft, Tesla and CD Projekt. Scores may also carry over poorly between companies.

## 5. Non-RAG control and RAG tracks

### 5a. Audit the news data (no API cost)

- **Coverage:** the cache has 653 scored days against 1,536 trading days, so at most about 43% are covered. Daily article counts cap at 25 with a median of 23, which suggests capped results bunched on certain days.
- **Other sources:** check whether the Alpaca and other local news data fill the gaps. Don't fetch new data unless explicitly asked.
- **Hindsight risk:** the LLM may already know how 2020-2025 events turned out. Compare scores with and without company names, and look at any period after the model's knowledge cutoff.

### 5b. Non-RAG control

Build the "per time frame" version from `README.md`, where the LLM sees all articles from a 5-day window. This shows whether RAG adds anything beyond giving the model more context.

### 5c. `news_features_from_rag`

- **Retrieval:** combine embeddings and keyword search. Only documents published before the prediction date can be retrieved, enforced by an automated test.
- **Features:** use the same schema as the current news features, so the tracks compare directly.
- **Comparisons:** fixed queries vs LLM-written queries, and today's news vs older context, which is RAG's real advantage over daily scoring.
- **Saved evidence:** store the retrieved articles for each day, and include the model and prompt in the cache key.
- **Pilot first:** try a sample of training days to measure quality and cost before running everything.

### 5d. `news_shaped_based_rag`

Retrieve similar past 30-day return patterns, in two stages:

1. **Shape only:** use what followed those past patterns. This is free and acts as a control.
2. **Shape plus news:** add the news from around those past dates.

If stage 2 beats stage 1, that's direct evidence for combining the two data types.

### 5e. Final comparison

Run this once, then unlock the test split for the final results.

| Question | How to measure it |
|---|---|
| Does retrieval find the right articles? | Precision against about 50 hand-labelled days, and whether earnings days are found |
| Do the features reflect real events? | Match with known events, and link to the size of the next move |
| Do the features improve prediction? | MSE and MAE against the baseline, direction against always-up, with confidence intervals |
| Is it practical? | Cost per day, and whether a prediction can be traced to its articles |

The project still has something to show if return prediction doesn't improve but RAG finds better evidence and detects events better. "RAG adds traceability but no predictive edge" is a legitimate finding.

## Open questions

1. Which research paper does the shape-based approach copy? Needed for 5d.
2. What's the API budget? This sets how much of step 4's news work and step 5 can run. The pilot will give a real cost per day.
3. Should move size be added as a second target? News may suit it better than direction. Direction and return prediction would stay as they are.

## Later, if useful

- Transformers, from `notes.txt`. They answer a different question from the one this plan tests.
- More target companies, which could add a lot of statistical power.
