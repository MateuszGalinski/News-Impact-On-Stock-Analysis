# Roadmap

Agreed plan as of 2026-11-07. The baseline work (steps 1, 2, 3a, 3c and 3d, recorded under "Completed work") is finished. The next step is A, then B. Nothing in steps B to G has been built yet.

**Project aim:** develop and empirically verify a method for combining structured data (prices) with unstructured data (news articles) that respects time, so that no information from the future leaks in. The method is a Time-Aware RAG, which only lets the model use information that was available at each point in time. The domain is market trend prediction for NVDA, from daily prices and financial articles.

**What the baseline is for:** the completed work shows that no model beats simple baselines on next-day NVDA returns, which matches the literature. That's the honest context for the thesis, not a problem to fix first. The project needs a fair baseline, not a strong one: built without leakage, scored on the same days, and different only in the input being tested. A finding such as "once leakage is controlled, news adds little or nothing" is a valid result.

**Rules for every step:**

- Keep the test split locked until the final comparison, and use it once.
- Fix the options to compare before seeing results, and report all of them.
- Judge directional accuracy against always-up, and MSE and R² against the train mean.
- Preserve the one-day news shift and the after-close roll, which prevent look-ahead leakage.
- Ask before any step that costs money (LLM or embedding calls), with a cost estimate first. Don't fetch new data unless explicitly asked.

## A. Frozen protocol

Fixed on 2026-11-07, before any results. Every comparison in steps B to G uses it.

- **Target:** next-day `LOG_CLOSE_RETURN`, with a 30-day window.
- **Evaluation:** walk-forward, 8 rounds, 781 scored days (2022-05-25 to 2025-07-08), common dates and standardization, using the first 90% of rows only. The 80/10/10 split is a second view. The test split is used once, in step G.
- **Models:**
  - RNN.
  - LSTM | Dropout 0.5.
  - XGBoost | Stronger, with a patience of 30.
  - Ridge, a linear regression on the same flattened windows. It shows whether the nonlinear models add anything over a straight line.
  - 3 seeds (42, 43, 44), averaged day by day. Ridge has no randomness.
- **References:** the train mean, and always-up for direction.
- **Metrics, all with stationary block-bootstrap intervals:**
  - MSE.
  - R² against the train mean.
  - Directional accuracy, reported as the gain over always-up.
  - Correlation between prediction and actual.
- **Decision rule:** a variant improves on what it's compared with if, for the same model on the same days, its directional-accuracy gain has a 95% interval above zero and MSE isn't clearly worse. R² and correlation are reported for every comparison but don't decide. A clear correlation gain without a direction gain is reported as a weaker sign, not a win.
- **Plots, one per cell:**
  - Predicted against actual over time, for all scored days and the last 120.
  - A scatter of predicted against actual.

  Expect nearly flat predictions against noisy actual returns. That's the honest picture.

**The comparisons the rule decides, fixed in advance:**

1. Simple news features vs no news: does news help at all?
2. Time-aware RAG vs no news, and vs simple news features: does RAG add anything?
3. Naive RAG vs time-aware RAG: here a better naive score isn't an improvement. It measures how much leakage inflates results.

## B. Reference run (no API cost)

Run the frozen models on the 4 existing input sets: no news, no news with indicators, news, and news with indicators. This needs Ridge added to `models.py`, and the four metrics and plots in one notebook. The result is the table every RAG variant is compared against.

## C. News data audit (no API cost)

- **Coverage:** the cache has 653 scored days against 1,536 trading days, so at most about 43% are covered. Daily article counts cap at 25, with a median of 23, which suggests capped results bunched on certain days.
- **Other sources:** check whether the Alpaca and other local news data fill the gaps.
- **Timestamps:** check that publish times are reliable enough for a hard time filter, since the whole method depends on them.

## D. Time-aware RAG (`news_features_from_rag`)

**What it adds:** today the LLM sees only one day's articles. With RAG, the query for day t also retrieves relevant older articles from everything published up to t, such as earlier coverage of the same product, lawsuit or earnings. That gives the LLM context it doesn't have now.

- **Index:** the existing GNews articles, embedded once. Each article keeps its timestamp, and its `trading_day` after the existing after-close roll.
- **Retrieval for day t:** a hard filter only allows articles available by the same cutoff the current news features use, so the one-day shift is preserved. The score is relevance weighted by recency, which is the time-aware part. Consider combining embeddings with keyword search.
- **LLM step:** the same output schema as the current news features, so only the context changes.
- **Cache:** keyed by date and variant, plus a hash of the prompt and model. The current cache is keyed by date alone and silently reuses stale rows.
- **Saved evidence:** store the retrieved articles for each day, so every prediction can be traced to its sources.
- **Checks, like `check_walk_forward`:** no retrieved article is ever after its cutoff, and changing future articles must not change any earlier feature.
- **Pilot first:** a sample of days to measure retrieval quality and the real cost per day, before the full run.
- **Optional control:** an LLM that sees all articles from a 5-day window, with no retrieval. It shows whether RAG adds anything beyond giving the model more context. Decide after the pilot, since it costs LLM calls too.

## E. Naive RAG: the leakage demonstration

The same pipeline as D with the time filter removed, so day t can retrieve articles from later dates. It's scored on the same days and clearly labelled as leaky. If it looks clearly better than time-aware RAG, that's direct evidence for the topic's central claim, that temporal control is needed.

## F. Leakage through the LLM's own memory

GPT already knows how NVDA did after each date, so even perfectly filtered retrieval can leak through the model itself (look-ahead bias, studied for GPT sentiment scores by Glasserman & Lin, 2023). On a sample of days, re-score with the company name and dates hidden, and compare with the normal scores. If the named version lines up much better with later returns, the model is using hindsight. Also look at any days after the model's knowledge cutoff.

## G. Final comparison and the locked test

All variants on the same 781 walk-forward days, then one run on the locked test split for the frozen models and chosen variants, reported whatever it shows.

| Question | How to measure it |
|---|---|
| Do the features improve prediction? | The decision rule in A, with all four metrics reported |
| How much does leakage inflate results? | Naive vs time-aware RAG (E), and named vs hidden company (F) |
| Does retrieval find the right articles? | Precision against about 50 hand-labelled days, and whether earnings days are found |
| Do the features reflect real events? | Match with known events |
| Is it practical? | Cost per day, and whether a prediction can be traced to its articles |

"Time-aware RAG adds traceability and removes leakage but no predictive edge" is a legitimate finding.

## Open questions

1. What's the API budget? It sets how much of D to F can run. The pilot in D will give a real cost per day.
2. Which research paper does the shape-based approach copy? Only needed if shape-based RAG is picked up later.

## Later, if useful

- **Range-based volatility with news (3d, steps 2-3):** the only target where prices clearly carry signal. Add the news features and an earnings-date flag. Set aside on 2026-11-07.
- **Feature composition test (3b):** which news feature groups matter, with a shuffled-news placebo.
- **5-day horizons (3c, stage 2).**
- **Pretraining on other companies (former step 4):** fit an LSTM on other semiconductor companies, then fine-tune on NVDA. Pretraining only earns its place if the fine-tuned model beats both NVDA-only versions, with the other companies' data ending before NVDA's scored period in each round. Other companies' news needs new LLM calls and a cache keyed by company as well as date.
- **Shape-based RAG (`news_shaped_based_rag`):** retrieve similar past 30-day return patterns, first by shape only (free, a control), then with the news from around those dates.
- **Transformers,** from `notes.txt`. They answer a different question from the one this plan tests.

# Completed work

The baseline steps, kept as the record. They were run before the plan above, so their models and metrics differ slightly from step A.

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

The learning curves showed overfitting, so this step tried to control it, then moved on to other targets.

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
- **Stronger XGBoost:** it has the best R², -0.0006 (95% interval -0.0043 to 0.0031) with no news and no features, but it's close to predicting the mean. Its spread is 0.05 of actual, about 92% of its predictions are up, and its correlation is at or below zero. That makes it a weak tool for testing features in a feature test. Its patience stopped it early, but the rerun below shows longer patience is worse.
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

If news wins, run a shuffled-news placebo and the news audit (step C) before trusting it. Article count could just reflect that big days attract more news, and the LLM may know how past events turned out.

**Stage 1 result (2026-11-06):** under the rules above, the test stops at question 1, and news shows no benefit.

- **Question 1:** HAR's R² is 0.012 (95% interval -0.020 to 0.051), so it doesn't pass. Its rank correlation is 0.133 (0.048 to 0.189), so there's a real but small pattern in which days are bigger. Unlike the return target, 13 of the 26 fitted setups, including HAR, have an R² above zero, but none has an interval above zero.
- **Question 2:** none of the 25 comparisons with HAR has an MSE interval above zero, and 6 are clearly worse. The 20-day average ranks days about as well as HAR, at 0.144.
- **Question 3:** in 13 comparisons, news is never better on MSE or rank correlation, and is worse on MSE in 5, including HAR + news vs HAR. It's better on MAE in 2 and worse in 2.
- **Big moves:** they aren't predicted. HAR predicted 0.022 for the largest move, 0.218, and no trained model catches more than about 6% of big moves.
- **Spikes:** checked after seeing results. The biggest 1% of days make up 33% of the train mean's squared error, but removing them only raises HAR's R² to 0.024, so they aren't hiding a stronger pattern.
- **80/10/10:** R² reaches 0.072 and rank correlation 0.36 over 123 days, but news is again never better.

These news features end two days before the scored move's close, and don't include scheduled events such as earnings dates, so this doesn't rule out better news inputs.

**Stage 2, 5-day horizons:** postponed on 2026-11-07, and only worth running if the range-based target in 3d works at 1 day. The plan was to run 5-day move size, as the average daily move over the 5 days, and 5-day return, each on 749 scored days. Those days overlap, so they're worth only about 150 independent periods. A 20-day horizon was left out, because about 31 independent periods is too few.

### 3d. Range-based volatility

Agreed on 2026-11-07, after a literature check. Change one thing at a time, so each result can be traced to its cause.

**What the literature says:**

- **Direction:** no credible study reliably predicts one stock's next-day direction or return from its own daily prices. Informal attempts like ours land at about chance. The successes use many stocks at once and rank them against each other, and their edge is small (for example Fischer & Krauss 2018, Gu, Kelly & Xiu 2020). Studies claiming 70-90% accuracy usually have a flaw such as look-ahead leakage.
- **Volatility:** it is predictable, and machine learning often beats HAR (Christensen, Siggaard & Veliyev 2023). Those studies measure volatility from intraday prices, which is much less noisy than the absolute daily return we used in 3c. News helped volatility forecasts most on jump days (Rahimikia & Poon).
- **How forecasts are judged:** with a noisy volatility measure, only some losses rank forecasts correctly. MSE and QLIKE are the standard ones (Patton 2011). Models are usually trained with MSE on the log of volatility, against a log-HAR baseline.

**Plan:** the range-based target first, then news on it, then many stocks. Only the first part was run before the project refocused on the RAG plan above.

1. **Range-based target.** Free, and uses the existing pipeline and models.
   - **Target:** the log of each day's range-based variance, using Garman–Klass plus the overnight gap, since earnings reactions mostly happen overnight. The open, high and low are already in the Stooq file.
   - **Input:** each day's value of the same measure, in place of the absolute return.
   - **Training:** unchanged, with MSE on the standardized target.
   - **Baseline:** log-HAR, fitted on the 1, 5 and 22-day averages of the log measure.
   - **Evaluation:** QLIKE and MSE, after converting forecasts back from logs to variance with a bias correction. Rank correlation is an extra check. MAE is dropped, because it isn't a robust loss for this.
   - **Price bands:** turn each forecast into a band around that day's real close for the next day, as `close × exp(±1.96 × forecast σ)`, a standard 95% band. The band is centred on today's price, because the target has no direction. Returns have more extreme days than a normal distribution, so expect coverage somewhat below 95%. On the scored days, report coverage (how many next-day closes land inside, against 95%) and average width, and compare with a fixed-width band from the train mean. A better model reaches the same coverage with narrower bands. Also check whether misses cluster on earnings days, and plot the bands against the real price.
   - **Question 1:** log-HAR must clearly beat the train mean, or news isn't tested on this target.
   - **Rules, fixed on 2026-11-07 before any results:** `training/benchmarking/notebooks/range_volatility.ipynb`, with the same 6 models, 4 input sets, 3 seeds and both evaluations as 3c. The train mean is the average variance of the days known before each round. The bias correction uses each fit's mean squared error on its known days, on the log scale. "Clearly beats" means the QLIKE improvement interval is above zero, and MSE and rank correlation aren't clearly worse. That applies to question 1 (log-HAR vs the train mean), question 2 (a trained model vs log-HAR) and question 3 (news vs no news with the same model, and log-HAR + news vs log-HAR). Yesterday's value and the 5-day and 20-day averages of the variance are extra baselines. The price bands are descriptive and don't decide anything. Earnings dates aren't in the data, so the largest band misses are listed for checking by hand.
   - **Results, run on 2026-11-07 (624 walk-forward fits, 78 on 80/10/10):**
     - **Question 1: fails by the fixed rule.** Log-HAR's QLIKE improvement over the train mean is 0.066 (−0.066 to 0.181), and no model's interval is above zero. So news isn't tested on this target as it stands.
     - **Why it fails is different from 3c.** Log-HAR ranks days well: rank correlation 0.52 (0.40 to 0.59), against 0.13 for move size. But 8 days (1%) make up 42% of its QLIKE, and the earnings jump on 2023-05-25 alone makes up about a quarter. Without those 8 days, log-HAR clearly beats the train mean on all three measures (QLIKE 0.107, 0.051 to 0.171). On the 80/10/10 validation days its QLIKE skill is 0.29 (0.04 to 0.50). At least 4 of the 8 days were earnings, which the inputs can't see coming.
     - **Question 2: no.** No trained model clearly beats log-HAR. 3 of 24 are clearly worse on QLIKE, and 18 of 24 on rank correlation. On the 80/10/10 validation days (2025-01-08 to 2025-07-08), none clearly beats it on QLIKE, but LSTM and LSTM | Dropout 0.5 with no news and no features are clearly better on rank correlation (+0.057, 0.008 to 0.129, and +0.075, 0.019 to 0.158). Walk-forward on the same days shows the same, so the LSTMs caught up with log-HAR in 2025 after trailing it in 2022-2024. It's 2 of 24 comparisons in one period, so a hint only.
     - **Question 3, shown but not a test: no sign news helps.** None of 13 is clearly better, and 3 are clearly worse on QLIKE. Log-HAR + news is level with log-HAR.
     - **Price bands:** log-HAR covers 93.9% of next-day closes at an average width of 12.2% of the price, against 94.8% at 13.3% for the train mean. The largest misses are the same shock days.

**Not done:** the overview chart shows QLIKE and rank correlation but not the MSE-based R², which is only in the tables. Add it if this target is picked up again.
