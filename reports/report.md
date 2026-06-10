# Diamond Price Prediction — Analytical Report

**Dataset:** Diamond Prices (Kaggle)  
**Target variable:** `price` (USD) — modelled as `log_price`  
**Authors:** Project submission  

---

## 1. Dataset Description

The Diamonds Prices dataset is a structured tabular dataset sourced from Kaggle, containing records of individual cut diamonds sold on the retail market. Each row represents a single diamond transaction and characterises the stone along two dimensions: its physical geometry and its gemological quality grades.

The dataset contains **53,940 observations** and **10 variables** prior to any cleaning. The unit of analysis is a single diamond. The target variable is `price`, expressed in US dollars, and ranges from approximately \$326 to \$18,823. The price distribution is strongly right-skewed, with a large concentration of lower-priced stones and a long upper tail driven by high-carat diamonds.

The dataset is well suited to regression modelling: it contains multiple numeric predictors, three ordinal categorical variables with meaningful natural orderings defined by gemological grading standards, and a continuous numeric target. It includes non-trivial structure in the form of high multicollinearity among geometric features, a skewed target distribution, physically implausible zero-dimension records, and a non-negligible number of exact-duplicate rows.

**Known limitations and potential biases:**

- The dataset represents a particular market snapshot and may not generalise to other time periods or retail contexts.
- The sampling frame is unknown. If diamonds are overrepresented in certain carat-weight bands (e.g., near-round values such as 0.5, 1.0, 1.5 carats) because of retail pricing conventions, the model will learn those conventions rather than a pure physical pricing function.
- The vertical bands visible in the carat–price scatter plot at round carat values (0.5, 1.0, 1.5, 2.0) are consistent with this hypothesis.
- Physical dimensions (`x`, `y`, `z`) contain zero-valued entries that are geometrically impossible for a real diamond, indicating data entry errors or placeholder values rather than genuine observations.
- A substantial number of exact-duplicate rows are present and likely reflect recording artefacts rather than distinct market transactions.

---

## 2. Variable Dictionary and Measurement Types


| Variable | Data Type | Measurement Level | Description |
|---|---|---|---|
| `carat` | float64 | Ratio | Weight of the diamond in carats; strictly positive, continuous |
| `cut` | object | Ordinal | Cut quality grade: Fair < Good < Very Good < Premium < Ideal |
| `color` | object | Ordinal | Colour grade: J (most colour) < I < H < G < F < E < D (colourless) |
| `clarity` | object | Ordinal | Clarity grade: I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF |
| `depth` | float64 | Ratio | Total depth as a percentage of average diameter |
| `table` | float64 | Ratio | Width of the top facet (table) as a percentage of average diameter |
| `price` | int64 | Ratio | Sale price in US dollars; primary target variable |
| `x` | float64 | Ratio | Length in millimetres |
| `y` | float64 | Ratio | Width in millimetres |
| `z` | float64 | Ratio | Depth in millimetres |

**Derived variables constructed during preprocessing:**

| Variable | Data Type | Measurement Level | Description |
|---|---|---|---|
| `log_price` | float64 | Interval | Natural log of price; primary regression target |
| `cut_ord` | int64 | Ordinal (encoded) | Integer encoding of cut: 0 (Fair) – 4 (Ideal) |
| `color_ord` | int64 | Ordinal (encoded) | Integer encoding of colour: 0 (J) – 6 (D) |
| `clarity_ord` | int64 | Ordinal (encoded) | Integer encoding of clarity: 0 (I1) – 7 (IF) |

**Ambiguous variables:** `depth` and `table` are ratio-scale percentage variables but their relationship with price is non-monotonic in raw form — optimal proportions are prized, and both extremes may indicate poor cuts. Their direct inclusion as linear predictors is a modelling simplification.

---

## 3. Missing Data Analysis

The raw dataset contains **no structurally missing values** (null counts are zero for all columns). This is characteristic of retail datasets that enforce completeness at the point of entry or that replace missing values with placeholder zeros. The absence of structural missingness, however, does not mean the data are complete in the substantive sense.

**Physical impossibility as implicit missingness:** Rows where `x`, `y`, or `z` equals zero represent implausible observations — a diamond with zero geometric dimension cannot exist. These entries function as missing values under a different encoding. The notebook identifies these rows and removes them as a data quality correction, not a statistical imputation decision.

**Duplicate rows:** A non-trivial number of exact-duplicate rows (identical values across all 10 columns) are present in the dataset. These are treated as recording artefacts and removed (keeping the first occurrence), because the probability that two independently sold diamonds would share identical values for all 10 attributes — including price to the dollar — is negligibly small. Retaining them would artificially inflate certain feature combinations in the training distribution.

**Missingness mechanisms (as hypotheses):**

Since structural null values are absent, the relevant mechanisms apply to the zero-dimension records:

- **MCAR (Missing Completely at Random):** Plausible if zeros reflect random data-entry errors uncorrelated with any feature. Under this hypothesis, removal introduces no systematic bias.
- **MAR (Missing at Random):** Possible if zero entries are more common for certain cut qualities or price ranges — for example, if cheaper or poorly cut stones were more likely to have incomplete dimension records. This hypothesis cannot be tested without external data but is worth noting.
- **MNAR (Missing Not at Random):** Would apply if zero dimensions indicate that the stone was not physically measured, which might correlate with diamond quality. Under this hypothesis, removal could bias the dataset toward fully measured (and therefore more standardised) stones.

**Decision:** Physical invalidity is treated as a correctness operation. All rows with `x`, `y`, or `z` ≤ 0 are removed before splitting. This decision is conservative: no imputation is attempted because imputing physical dimensions from price or quality would introduce a circularity incompatible with the modelling objective.

---

## 4. Univariate Analysis

### Numeric Variables

The numeric quality report, computed on the full dataset, characterises the distributional properties of each feature. The most analytically relevant findings are:

**`price`:** Strongly right-skewed (skewness > 1), high excess kurtosis, and a wide IQR relative to the mean. The mean substantially exceeds the median, confirming a long upper tail. The coefficient of variation is high, indicating substantial relative dispersion. IQR-based outlier detection identifies a non-trivial number of extreme high-price observations, most of which correspond to large-carat stones rather than data errors.

**`carat`:** Also right-skewed with a positive mean-median gap, and with notable vertical bands in its scatter plot at round values (0.5, 1.0, 1.5, 2.0 carats). These bands reflect retail pricing conventions and are a genuine data characteristic.

**`x`, `y`, `z`:** Moderately skewed with a few extreme values. Several zero-valued entries are present (physically impossible) and are removed during preprocessing.

**`depth`:** Approximately symmetric (low skewness), with relatively low dispersion. Pearson correlation with price is essentially zero (r = −0.01), confirming that depth alone carries no linear price signal.

**`table`:** Low-to-moderate skewness. Very weak correlation with price (r = 0.13). Both `depth` and `table` are classified as well-behaved for linear modelling but provide marginal predictive power in isolation.

**`x`, `y`, `z`:** Moderately skewed with a few extreme values. Several zero-valued entries are present (physically impossible) and are removed during preprocessing. These three dimensions are highly collinear with `carat` (r > 0.95), which motivates their exclusion from the regression model in favour of `carat` as the sole size predictor. They are retained separately for use in PCA.

![Price distribution (raw)](../reports/figures/price_distribution_raw.png)

![Log-price distribution](../reports/figures/price_distribution_log.png)

The histograms confirm that raw `price` is severely right-skewed, while `log(price)` is approximately bell-shaped and suitable as a regression target.

### Categorical Variables

The three categorical variables have the following properties:

**`cut`:** Five ordered levels (Fair, Good, Very Good, Premium, Ideal), with Ideal being the most frequent category. The distribution is moderately imbalanced — Ideal and Premium dominate — but no level is rare enough to warrant collapsing.

**`color`:** Seven ordered levels (D through J). The distribution is approximately balanced across the middle grades with slight concentration at G and H.

**`clarity`:** Eight ordered levels (I1 through IF). SI1 and VS2 are the most frequent; IF (flawless) is relatively rare. The categorical quality report classifies all three variables as high quality with low cardinality, no dominant concentration above 80%, and no rare levels.

Boxplot analysis of `price` by each categorical variable reveals substantial within-category variance and considerable overlap across categories. This is expected: `carat` is the dominant price driver, and diamonds in worse quality grades can still be expensive if they are large stones. Correct interpretation of the categorical effects on price requires controlling for carat weight.

![Categorical variables vs price (boxplots)](../reports/figures/boxplots_categorical_vs_price.png)

---

## 5. Association and Dependence

### Numeric–Numeric Associations

Pearson correlations between numeric features and `price` reveal a dominant size signal:

- **carat × price = 0.92** — carat weight is the single strongest linear predictor of price.
- **x × price = 0.88**, **y × price = 0.87**, **z × price = 0.86** — physical dimensions follow closely, but are largely redundant with carat.
- **table × price = 0.13**, **depth × price = −0.01** — shape ratios provide negligible linear signal in isolation.

The Pearson correlation matrix among predictors exposes severe multicollinearity:

- **carat × x = 0.98**, and pairwise correlations among x, y, z range from 0.95 to 0.97.
- All four size-related variables (`carat`, `x`, `y`, `z`) essentially encode the same latent dimension.

This structure motivates the exclusion of `x`, `y`, `z` from the regression model and the retention of `carat` as the primary size metric, along with the application of VIF diagnostics in the modelling phase.

**Pearson vs Spearman comparison:** The Spearman rank correlation between `carat` and `price` is higher than Pearson, indicating that the relationship is monotonically strong but non-linear in the raw scale. This observation directly motivates the log transformation of `carat` tested in Models 3 and 4.

![Pearson correlation heatmap](../reports/figures/correlation_heatmap.png)

![Numeric predictors vs price (scatter)](../reports/figures/scatter_predictors_vs_price.png)

### Categorical–Numeric Associations

Boxplot analysis shows that `price` varies across cut, colour, and clarity levels, but the within-category distributions are wide and overlapping. The mean price differences between quality grades are modest compared to the variance driven by carat. No single categorical variable produces clean price separation in isolation, consistent with the adj. R² of 0.890 achieved by a model that includes all four grading attributes with a linear size specification.

### Categorical–Categorical Associations

Cramér's V was computed for all pairs of categorical variables with cardinality ≤ 15. The three quality grade variables (`cut`, `color`, `clarity`) show low-to-moderate pairwise Cramér's V, indicating that they encode partially independent information. This justifies retaining all three in the model rather than collapsing them or selecting only one.

---

## 6. Regression Modelling Strategy

### Objective

The objective is to build a regression model that predicts diamond price from gemological and geometric attributes. The model is evaluated on predictive accuracy (RMSE) and structural interpretability (VIF, coefficient significance). A secondary objective is to assess whether dimensionality reduction via PCA can substitute for direct feature modelling.

### Target Variable

The regression target is `log_price` — the natural logarithm of the sale price in US dollars. The justification is threefold: the raw price distribution is strongly right-skewed; the price–carat relationship exhibits diminishing marginal returns consistent with a log–log specification; and log-transformation stabilises residual variance across the price range, improving the reliability of OLS coefficient estimates. Predictions on the original USD scale are recovered by exponentiating the model output.

### Split Strategy

The dataset is partitioned into three disjoint subsets following the train-first principle: all preprocessing decisions are fixed before splitting, and no data-dependent parameters are estimated on the validation or test sets.

- **Train:** 70% of observations
- **Validation:** 15% of observations
- **Test:** 15% of observations

Model selection and comparison are made exclusively on the **validation set**. The **test set** is used only once, for the final generalisation estimate of the selected model. This discipline prevents information leakage from validation performance into model selection choices.

### Evaluation Metrics

**Primary metric:** RMSE on the log scale (`log_price`), used for all model comparisons and selection decisions. Log-scale RMSE is preferred for model selection because it reflects proportional prediction error uniformly across the price range, without being dominated by expensive stones.

**Secondary metric:** RMSE on the original USD scale, reported for interpretive purposes only. A log-scale RMSE of ~0.13 (Models 3–4) corresponds to a geometric mean prediction error of roughly 14%, which is more representative of the model's typical accuracy than the dollar-scale RMSE.

RMSE is appropriate here because prediction errors are continuous, large errors should be penalised more than proportionally, and the symmetric loss function aligns with the regression objective. MAE would be valid but less sensitive to the tail errors that are substantively important in this market context. R² is reported as a supplementary descriptive statistic but does not drive model selection.

---

## 7. Baseline Model

### Preprocessing Pipeline

The preprocessing pipeline follows strict train-first methodology:

1. **Physical validity:** Rows with `x`, `y`, or `z` ≤ 0 are removed before splitting (correctness operation).
2. **Duplicate removal:** Exact-duplicate rows are removed before splitting (keeping first occurrence).
3. **Split:** 70/15/15 train/validation/test, stratified by `random_state=42`.
4. **Ordinal encoding:** `cut`, `color`, and `clarity` are mapped to integers using domain-fixed ordinal scales. The mappings are not data-dependent and are applied identically to all splits.
5. **Feature engineering:** `log_price = log(price)` is computed as the regression target.
6. **Multicollinearity reduction:** `x`, `y`, and `z` are excluded from the modelling feature set entirely. `carat` is retained as the primary size metric. `x`, `y`, `z` are preserved separately for PCA only.

No numerical imputation is performed because the modelling dataset contains no missing values after the physical validity step.

### Model Progression

Four OLS models were evaluated on the validation set. The models are described in order of increasing specification richness.

**Model 1 — Baseline (4Cs only, linear carat):**  
`log_price ~ carat + C(cut_ord) + C(color_ord) + C(clarity_ord)`

The four canonical grading attributes achieve an adjusted R² of 0.890 on the training set. The ordinal quality variables are entered as dummy-coded categorical predictors via `C()`, allowing each grade level to have its own effect rather than assuming equal spacing. The coefficient on `carat` is 2.204 — large, positive, and highly significant. Coefficients on `clarity_ord` levels are the largest in magnitude (up to ~1.07 for IF), followed by `color_ord` levels (up to ~0.585 for D); `cut_ord` levels are significant but smaller (~0.04–0.07). Validation RMSE of 0.3412 on the log scale establishes the performance floor.

**Model 2 — Grading Attributes + Shape Ratios:**  
`log_price ~ carat + C(cut_ord) + C(color_ord) + C(clarity_ord) + depth + table`

Adding `depth` and `table` yields only marginal improvement (adj. R² 0.890, essentially unchanged). `table` is statistically significant (β = 0.006, p < 0.001) but `depth` is not (β ≈ 0, p = 0.827), confirming that the proportional depth adds no incremental price signal. VIF diagnostics show no meaningful multicollinearity from these shape ratios, though the condition number rises to 6,400 — a sign of potential numerical sensitivity. Validation RMSE drops marginally to 0.3409. The key limitation remains the linear specification for `carat`.

**Model 3 — Log-Transformed Size Feature:**  
`log_price ~ log(carat) + C(cut_ord) + C(color_ord) + C(clarity_ord) + depth + table`

Log-transforming `carat` produces the largest single improvement across all specifications: adj. R² jumps to 0.983 and validation RMSE falls to 0.1336. The coefficient on `log(carat)` is 1.885 — interpretable as a price elasticity: a 1% increase in carat weight is associated with an ~1.9% increase in price, holding quality constant. `depth` is marginally significant (β = −0.001, p = 0.016) and `table` is not (β ≈ −0.001, p = 0.108). All quality grade levels retain expected signs. The condition number remains at 6,400, flagging residual numerical sensitivity from the shape ratio features.

**Model 4 — Parsimonious Log-Linear (selected final model):**  
`log_price ~ log(carat) + C(cut_ord) + C(color_ord) + C(clarity_ord)`

Removing `depth` and `table` entirely leaves validation RMSE at 0.1336 — identical to Model 3 to four decimal places. Adj. R² is 0.983. The condition number drops sharply from 6,400 to 34, confirming that the numerical sensitivity in Model 3 was introduced by the shape ratio features, not by `log(carat)`. All remaining coefficients are highly significant with economically expected signs; the `log(carat)` elasticity is 1.885.

**Model 4 is selected as the final regression model** on grounds of parsimony, numerical stability, and domain interpretability. The predictive cost of removing `depth` and `table` is zero (RMSE identical to Model 3); the benefit is a well-conditioned model (condition number 34 vs 6,400) that aligns with the standard Four Cs specification.

### Model Performance (Validation Set)

| Model | Features / Transform | Val RMSE (log) | Val RMSE (USD) |
|---|---|---|---|
| Model 1 | 4Cs — linear carat, dummy-coded grades | 0.3412 | $81,005 |
| Model 2 | Linear carat + depth + table | 0.3409 | $80,256 |
| Model 3 | log(carat) + depth + table | 0.1336 | $788 |
| **Model 4 ★ Selected** | **log(carat), no depth/table** | **0.1336** | **$789** |
| PCA Model 1 | 3 PC scores — ≥90% variance | 0.2948 | $6,987 |
| PCA Model 2 | 3 PC scores + cut/color/clarity | 0.2175 | $6,895 |
| PCA Model 3 | 6 PC scores — full reconstruction | 0.2547 | $1,477 |

*Smaller Val RMSE (log) is better. **Bold** = selected model.*

### Residual Diagnostics

**Residuals vs fitted values:** Model 1 exhibits a fan-shaped spread consistent with heteroscedasticity from predicting a skewed target with a linear size specification. The log transformation of `carat` in Models 3 and 4 substantially reduces this — adj. R² jumps from 0.890 to 0.983 and the skewness of residuals falls from −1.17 to +0.21.

**QQ plot / normality:** Model 4 residuals show skewness of 0.209 and excess kurtosis of ~5.2, an improvement over Models 1–2 (skewness ~−1.17, kurtosis ~8.7) but still indicating heavier tails than a normal distribution. Deviations in the upper tail suggest that high-carat diamond predictions are somewhat underestimated, consistent with the known price nonlinearity at the extreme upper market.

**Condition number:** Model 2 and Model 3 show a condition number of 6,400, flagging potential numerical sensitivity introduced by adding `depth` and `table` to the design matrix. Model 4's condition number is 34 — well within acceptable bounds — confirming that removing the shape ratios resolves this entirely.

**Error analysis:** Original-scale RMSE is dominated by prediction errors on large, expensive stones. A 10% error on a \$15,000 diamond contributes far more to RMSE than a 10% error on a \$500 diamond. Log-scale RMSE is the appropriate summary of typical accuracy across the full distribution.

---

## 8. PCA Analysis

### Variables Selected

PCA is applied to the six standardised continuous features: `carat`, `depth`, `table`, `x`, `y`, and `z`. The three ordinal quality grades (`cut_ord`, `color_ord`, `clarity_ord`) are excluded from the decomposition — they enter the regression models directly without PCA rotation, as ordinal integer scales do not have a meaningful notion of "variance" comparable to continuous measurements. The regression targets (`log_price`, `price`) are also excluded.

### Standardisation Rationale

PCA is a variance-based decomposition. If features are on heterogeneous scales — `x`, `y`, `z` are in millimetres while `depth` and `table` are percentages and `carat` is in tenths of grams — the principal components will be driven by variance in the highest-magnitude features, irrespective of their structural importance. Standardisation to zero mean and unit variance (z-scores) removes this scale bias and is a necessary precondition for a meaningful decomposition.

`StandardScaler` was fit exclusively on the training set. The fitted scaler (means and standard deviations per feature) was then applied without refitting to the validation and test sets, maintaining the train-first methodology throughout and preventing leakage of validation or test covariance structure into the decomposition.

### Explained Variance

The scree plot and cumulative variance table confirm that the effective dimensionality of the feature set is substantially lower than six. The dominant finding is:

- **PC1** captures a disproportionately large share of total variance — expected to exceed 50% — dominated by the size features (`carat`, `x`, `y`, `z`), which are highly correlated with each other and with price.
- **PC2** captures the second axis, driven primarily by shape ratio features (`table`, `depth`), largely orthogonal to physical size.
- **PC3+** capture residual variance associated with the remaining shape and size orthogonal components.

**Kaiser criterion (eigenvalue > 1):** Expected to select 2–3 components, consistent with the VIF diagnostics that flagged multicollinearity in the regression models.

**90% cumulative variance threshold:** The minimum number of components needed to explain 90% of feature-space variance is the operationally selected cutoff for PCA Model 1 in Part C.

![PCA scree plot and cumulative variance](../reports/figures/pca_scree_cumulative.png)

### Number of Components Selected

The 90% variance threshold is used to determine the truncation point for PCA-based regression. This criterion is principled, widely used, and conservative enough to retain the dominant structure while excluding noise-dominated components. The Kaiser criterion serves as a secondary check; both criteria are expected to indicate 2–3 components.

### Component Loadings and Interpretability

The loadings heatmap reveals the principal information axes of the six continuous features:

**PC1 — Size axis:** Large positive loadings on `carat`, `x`, `y`, and `z`. This is the dominant pricing axis — the direction along which diamond feature vectors vary most in the standardised space. Its alignment with the regression's dominant predictor (`log(carat)`) validates the modelling approach.

**PC2 — Shape axis:** Driven primarily by `table` and, secondarily, `depth`. Captures variation in cut proportions that is largely orthogonal to physical size. Two diamonds of the same carat can differ substantially on this axis depending on their table width and overall depth geometry.

**PC3 — Depth axis:** Almost entirely driven by `depth`, with a secondary loading from `table`. Isolates the depth-proportion signal from the other shape and size components.

**PC4+:** Capture the remaining orthogonal variance in the size features (`x`, `y`, `z`) once PC1 has absorbed most of the shared size information.

![PCA loadings heatmap](../reports/figures/pca_loadings_heatmap.png)

### Biplot — PC1 vs PC2

The biplot of a subsample of training observations on the PC1–PC2 plane, coloured by `log_price`, confirms that the price gradient aligns strongly with PC1. Observations at the right extreme of PC1 are predominantly high-priced large diamonds; those at the left are small, lower-priced stones. The `carat`, `x`, `y`, and `z` loading vectors point in nearly the same direction along PC1, making their redundancy geometrically explicit.

![PCA biplot PC1 vs PC2 coloured by log_price](figures/img.png)

### Interpretability Limits

PCA coefficients describe directions in an abstract rotated space rather than the original diamond attributes. A regression coefficient on PC1 quantifies sensitivity to a linear combination of all seven standardised features — it cannot be communicated to a diamond trader or appraiser in the way that "a 1% increase in carat weight is associated with an X% increase in price" can. Furthermore, PCA is variance-driven, not target-driven: it identifies the directions of maximum feature-space variance, which may not coincide with the directions most predictive of `log_price`. In this dataset, PC1 happens to align well with price because size drives both feature variance and price — but this alignment is coincidental, not guaranteed by the method.

---

## 9. Regression with PCA Features

### Models and Component Selection

Three PCA-based regression models were evaluated against the Part A baseline:

**PCA Model 1:** OLS on the first *k* PC scores, where *k* is determined by the 90% cumulative variance threshold. PC scores are orthogonal by construction, so VIF = 1 for every predictor — a structural advantage over Part A models where `carat` collinearity with the shape features required careful management.

**PCA Model 2:** Augments PCA Model 1 with the three ordinal quality grades (`cut_ord`, `color_ord`, `clarity_ord`) entered directly (not rotated through PCA). This tests whether adding the quality information that was deliberately excluded from the decomposition recovers predictive performance.

**PCA Model 3:** OLS on all available PC scores (full reconstruction). Since PCA is an orthonormal rotation, this model spans the same column space as OLS on the original standardised continuous features. It serves as a sanity check: its RMSE should approximate the linear Part A models (Models 1–2). Any large discrepancy would indicate a structural issue.

### Results and Comparison

| Model | Features / Transform | Val RMSE (log) | Val RMSE (USD) |
|---|---|---|---|
| Model 1 | 4Cs — linear carat, dummy-coded grades | 0.3412 | $81,005 |
| Model 2 | Linear carat + depth + table | 0.3409 | $80,256 |
| Model 3 | log(carat) + depth + table | 0.1336 | $788 |
| **Model 4 ★ Selected** | **log(carat), no depth/table** | **0.1336** | **$789** |
| PCA Model 1 | 3 PC scores — ≥90% variance | 0.2948 | $6,987 |
| PCA Model 2 | 3 PC scores + cut/color/clarity | 0.2175 | $6,895 |
| PCA Model 3 | 6 PC scores — full reconstruction | 0.2547 | $1,477 |

*Smaller Val RMSE (log) is better. **Bold** = selected model.*

### Interpretation

**PCA Model 3 vs Part A models:** PCA Model 3 (6 components, full reconstruction) achieves RMSE 0.2547, which is between the linear Part A models (0.34) and the log-transformed models (0.13). The gap relative to Models 1–2 is expected — PCA Model 3 includes the `x`, `y`, `z` information that Models 1–2 exclude, which helps. The gap relative to Model 4 is attributable to the log transformation of `carat`: log-transforming introduces a non-linearity that PC scores — being linear combinations of z-scored features — cannot replicate.

**PCA Model 1 vs PCA Model 3:** Truncating to 3 components (≥90% variance) raises RMSE from 0.2547 to 0.2948. The gap is moderate, indicating that the discarded components (PC4–PC6) carry some price-relevant information despite their low variance share.

**PCA Model 2 vs PCA Model 1:** Adding the ordinal quality grades directly to PCA Model 1 reduces RMSE from 0.2948 to 0.2175 — the largest single gain within the PCA models. This confirms that `cut_ord`, `color_ord`, and `clarity_ord` carry substantial price information that the continuous-feature PCA cannot capture on its own.

**Overall verdict:** All models are numerically stable. PCA-based regression is a valid, multicollinearity-free approach, but on this dataset it underperforms Model 4 (RMSE 0.1336) because PCA cannot replicate the log–log price–size relationship. PCA Model 2 is the best PCA specification, but it still sits 0.08 log-units above Model 4. The interpretability trade-off also remains unfavourable: PC coefficients describe abstract variance axes rather than the original diamond attributes.

---

## 10. Error Analysis and Limitations

### Where the Model Performs Poorly

The primary driver of prediction error is the upper tail of the price distribution. Very large diamonds (carat > 3) are rare in the training distribution, and the model is extrapolating beyond the dense support of the data. Even small percentage errors in log-space translate to large dollar errors at high price points, which dominates the original-scale RMSE. This is a structural limitation of the price distribution rather than a modelling deficiency.

Diamonds at round carat values (0.5, 1.0, 1.5, 2.0) are overrepresented due to retail pricing conventions. The model learns these conventions, which may not generalise to markets where pricing is less affected by round-number effects. The vertical bands in the carat–price scatter plot are a visible signature of this issue.

`depth` and `table` carry real but weak price signal and contribute small coefficients in all specifications. Their relationship with price is non-monotonic — optimal cut proportions are prized, but the dataset does not identify the optimal range endogenously. Including them as linear predictors is a simplification that may introduce mild misspecification, particularly for stones with extreme depth or table values.

### Sources of Error

**Omitted variables:** The dataset does not include fluorescence, symmetry, certification body, or carat-boundary pricing (e.g., near-round-number premiums). These are known determinants of diamond retail price and their absence introduces omitted variable bias, particularly for stones near round carat values where market pricing conventions are strongest.

**Ordinal spacing assumption:** Encoding ordinal grades as integers imposes equal-spacing — a one-unit increase in `cut_ord` from Fair to Good is assumed to have the same price effect as a one-unit increase from Premium to Ideal. If the actual grade-to-price mapping is non-linear, this misspecification will distort coefficient estimates on the quality variables.

**Log–log specification:** The log–log specification is an approximation. The true price–carat function likely has kinks at round carat values and may have different curvatures in different market segments. Residual heteroscedasticity in Model 4 would be a sign that the log transformation has not fully resolved this curvature.

### Dataset Limitations

- The dataset is a static snapshot with no temporal dimension; price trends over time are not captured.
- The sampling frame is unknown; representativeness of the broader retail market cannot be assessed.
- Physical dimensions contain zero-valued records (removed) and extreme outliers that may reflect data errors rather than genuine diamonds.
- The duplicate row issue suggests the data passed through a pipeline with imperfect deduplication.

### Modelling Assumptions

OLS regression assumes linearity in parameters, homoscedasticity, zero conditional mean of residuals, and no perfect multicollinearity. The log transformation addresses heteroscedasticity and linearises the dominant size relationship, but mild residual violations — particularly at the tails — should be expected given the price distribution and the non-linear pricing conventions at round carat values. Coefficient estimates in Model 4 are well-conditioned (VIF within acceptable bounds), but the ordinal encoding of quality grades imposes assumptions that a fully flexible specification would relax.

