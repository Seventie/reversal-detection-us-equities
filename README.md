# Detecting Reversal Points in US Equities 📈

**Kaggle Competition**: [Detecting Reversal Points in US Equities](https://www.kaggle.com/competitions/detecting-reversal-points-in-us-equities)

A machine learning approach to predict local highs and lows (reversal points) in US equity markets using technical analysis features and advanced feature engineering.

---

## 🎯 Problem Statement

### What is a Reversal Point?

A **reversal point** is a local minimum (support/trough) or maximum (resistance/peak) in a stock's price trajectory where the price direction changes. Detecting these points is one of the most valuable yet challenging problems in quantitative finance because:

- **High Reward**: Reversals are high-probability trading opportunities with favorable risk-reward ratios
- **High Difficulty**: They only become obvious in hindsight; predicting them in real-time is extremely hard

### Competition Objective

Given anonymized OHLCV (Open, High, Low, Close, Volume) data from multiple US equities over 2023-2025, build a classification model to:
1. Identify which price points are **true reversal points** (local extrema)
2. Distinguish them from **random price movements** (noise)
3. Predict reversals **in real-time** (without future data)

**Evaluation Metric**: F1 Score (balances precision and recall, crucial for imbalanced reversal data)

---

## 🔴 Core Challenges We're Facing

### 1. **The Definition Problem**
- **The Issue**: What qualifies as a "reversal"? A 2% move? 5%? Over how many candles?
- **Why It's Hard**: Without a clear definition, you can't label training data. Different definitions lead to entirely different models.
- **Current Impact**: Our `class_label` column is mostly NaN because we haven't finalized the reversal definition.

### 2. **Extreme Class Imbalance**
- **The Issue**: Reversals are rare. In typical equity data:
  - ~85% of points are normal trading (no reversal)
  - ~15% might be potential reversals
  - True high-conviction reversals: <5% of data
- **Why It's Hard**: A naive model predicting "no reversal" everywhere gets 85%+ accuracy but is useless
- **Current Impact**: Standard accuracy metrics are misleading; we need F1, precision-recall focus

### 3. **Data Leakage: The Causality Trap**
- **The Issue**: Technical indicators like `peaks_{X}` and `troughs_{X}` require looking backward *or forward* in data
- **Why It's Hard**: If we accidentally use future data to calculate features, our model will show unrealistic 90%+ performance during backtesting but fail in live trading
- **Current Impact**: With 1,982 features, it's easy to hide lookahead bias. Need strict feature validation.

### 4. **Market Regime Changes**
- **The Issue**: Patterns that work in bull markets fail in bear markets. Fed policy, macro conditions, volatility environments all change reversal dynamics
- **Why It's Hard**: Training on 2023-2024 bull market data often fails on 2025 regime changes
- **Current Impact**: Our dataset spans different regimes; need to validate across all periods

### 5. **Multiple Asset Heterogeneity**
- **The Issue**: Different stocks (ticker_id 1-6) behave completely differently
  - Tech stocks revert more frequently
  - Utilities are stickier
  - Micro-caps have different dynamics than large-caps
- **Why It's Hard**: One-size-fits-all model often fails; need regime-aware or per-stock models
- **Current Impact**: Need careful cross-validation by stock ID

### 6. **The Curse of Dimensionality**
- **The Issue**: We have ~1,982 features but only ~10K rows
- **Why It's Hard**: With 5 rows per feature, we're in high-dimensional space where overfitting is almost guaranteed
- **Current Impact**: Model will likely memorize training data rather than learn generalizable patterns

### 7. **The Real-Time Prediction Dilemma**
- **The Issue**: Predicting *when* a reversal happens requires predicting X bars ahead, but X is unknown:
  - Predict 1 bar ahead: Minimal profit opportunity
  - Predict 5 bars ahead: Much harder
  - Predict 20 bars ahead: Essentially predicting prices (nearly impossible)
- **Why It's Hard**: There's a fundamental tradeoff between prediction horizon and accuracy
- **Current Impact**: Our features don't explicitly define this horizon

### 8. **Market Efficiency / Rational Pricing**
- **The Issue**: Efficient Market Hypothesis suggests reversal patterns shouldn't exist or should be instantly arbitraged away
- **Why It's Hard**: Any patterns we find might be statistical noise rather than real inefficiencies
- **Current Impact**: Our baseline might be barely better than random

---

## 🚀 Our Current Approach

### Phase 1: Comprehensive Feature Engineering (Current)

We're creating 1,982+ technical indicators across multiple dimensions:

#### **Core Momentum Features**
```
- ratio: Price relative change metric
- momentum: Rate of price change
- sm_momentum: Smoothed momentum (noise reduction)
- sm_ratio: Smoothed ratio
```

#### **Threshold-Based Features** (`{X}` = varying thresholds like 99.0, 99.5, 100.0, etc.)
```
- cross_threshold_from_above_{X}: Detects price crossing support from above
- cross_threshold_from_below_{X}: Detects price crossing resistance from below
```

#### **Trend-Zone Features**
```
- trending_up_and_below_{X}: Price trending up but still below level X (bullish signal)
- trending_down_and_above_{X}: Price trending down but still above level X (bearish signal)
```

#### **Technical Structure Detection**
```
- peaks_{X}_min/max: Local highs near threshold X
- troughs_{X}_min/max: Local lows near threshold X
- zone_{X}_min/max: Price staying within range X
```

### Why This Approach?

**Rationale**: Rather than manually selecting 5-10 indicators, we're creating a **rich feature space** that:
- Captures multiple dimensions of reversal patterns simultaneously
- Allows the ML model to discover which combinations matter
- Tests multiple thresholds to find optimal support/resistance levels
- Creates both min and max features for capturing asymmetries

### Data Organization

```
Columns:
├── Identifiers
│   ├── ticker_id: Stock ID (1-6)
│   ├── train_id: Training batch ID
│   └── t: Date (2023-2025)
├── Basic Metrics
│   ├── ratio, momentum, sm_momentum, sm_ratio
├── 1,970+ Technical Features
│   ├── Threshold crossings (multiple {X} values)
│   ├── Trend patterns (trending_up_and_below, trending_down_and_above)
│   ├── Peaks/Troughs detection
│   └── Support/Resistance zones
└── Target (Currently NaN, to be filled)
    └── class_label: 0=No reversal, 1=Reversal point

Rows: ~10,000 price points across 6 stocks over 3 years
```

---

## 🛣️ Roadmap & Next Steps

### Immediate (This Week)
- [ ] **Define Reversal Target**: Establish clear mathematical definition
  - Determine lookback window (e.g., 5 bars back)
  - Determine lookahead window (e.g., 3 bars forward for confirmation)
  - Set minimum move threshold (e.g., 2% price change)
- [ ] **Label Training Data**: Create `class_label` using definition
- [ ] **Feature Validation**: Audit 1,982 features for lookahead bias
  - Ensure all features use only historical data
  - Document each feature's lookback window

### Short-term (Weeks 2-3)
- [ ] **Feature Selection**: Reduce 1,982 → 50-200 meaningful features
  - Use correlation analysis to remove redundant features
  - Use mutual information / feature importance to identify predictive features
  - Validate selected features across different stock regimes
- [ ] **Handle Class Imbalance**: 
  - Implement weighted loss functions
  - Try SMOTE for synthetic minority oversampling
  - Evaluate per-class F1 scores
- [ ] **Train Baseline Models**:
  - LightGBM (fast, interpretable)
  - XGBoost (strong performance)
  - Logistic Regression (benchmark)

### Medium-term (Weeks 4-6)
- [ ] **Cross-Validation Strategy**: 
  - Time-based splits (ensure no data leakage)
  - Per-stock validation (ensure per-asset robustness)
  - Regime-based validation (bull market, bear market, sideways)
- [ ] **Hyperparameter Tuning**: Optimize for F1 score
- [ ] **Error Analysis**: 
  - When does model fail? (certain stocks? regimes?)
  - False positives vs false negatives tradeoff

### Long-term (Weeks 7-8)
- [ ] **Advanced Models**: Try LSTMs / GRUs for temporal patterns
- [ ] **Ensemble Methods**: Combine multiple models
- [ ] **Live Backtesting**: Simulate real-time predictions with proper data leakage safeguards
- [ ] **Competition Submission**: Final predictions on test set

---

## 📊 Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| **F1 Score (Test)** | > 0.65 | Top competitors likely 0.70+ |
| **Precision** | > 0.60 | Minimize false reversals (costly in trading) |
| **Recall** | > 0.70 | Catch most true reversals |
| **Per-Stock F1** | Consistent | No single stock should dominate |
| **Generalization** | Minimal drop | CV F1 ≈ Test F1 (avoid overfitting) |
| **Feature Purity** | No lookahead | All features use only historical data |

---

## 🔧 Technical Stack

- **Language**: Python 3.8+
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, pandas, numpy
- **Data Processing**: Jupyter Notebooks
- **Evaluation**: scikit-learn metrics (F1, precision, recall, confusion matrix)
- **Versioning**: Git + GitHub

---

## 📂 Project Structure

```
reversal-detection-us-equities/
├── README.md                          # This file
├── notebooks/
│   └── newdata-us-reversal-1.ipynb   # Current feature engineering work
├── data/
│   └── .gitkeep                       # Data folder (gitignored in practice)
├── src/
│   ├── feature_engineering.py         # Feature calculation functions
│   ├── data_loader.py                 # Data loading & preprocessing
│   └── model_utils.py                 # Model training & evaluation
├── results/
│   └── .gitkeep                       # Model outputs & predictions
└── requirements.txt                   # Python dependencies
```

---

## 🤝 Key Insights & Lessons Learned

### What We Know Works
1. **Multiple thresholds beat single thresholds**: Testing multiple support/resistance levels captures more patterns
2. **Smoothed indicators help**: sm_momentum and sm_ratio reduce noise
3. **Combining indicators > Single indicators**: Reversal patterns are multi-dimensional

### What We're Still Figuring Out
1. **Optimal threshold values**: {X} parameter space is huge
2. **Ideal feature count**: 1,982 is likely too many; need systematic reduction
3. **Per-stock vs global models**: Should we build 6 separate models or one unified model?
4. **Lookback window for features**: How much history is enough without overfitting?

---

## ⚠️ Risk Factors & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Data Leakage** | Models fail in live trading | Strict feature audit; time-based CV only |
| **Overfitting** | 1,982 features on 10K rows | Feature selection; regularization; early stopping |
| **Class Imbalance** | Model predicts "no reversal" always | Weighted loss; F1 metric focus; SMOTE |
| **Regime Changes** | Past patterns don't predict future | Validate on multiple market periods |
| **Asset Heterogeneity** | One model doesn't fit all stocks | Cross-validate per ticker; consider per-stock models |
| **Definition Ambiguity** | Garbage labels → garbage predictions | Clear, testable reversal definition needed |

---

## 📖 References & Further Reading

### Academic Papers
- "Enhancing stock market trend reversal prediction using feature engineering and deep learning" (2024) - NIH/PMC
- "A new LSTM based reversal point prediction method" (2020) - LSTM approaches to reversal detection
- Research shows US equity reversals are harder to predict (55.2% F1) vs Chinese markets (68.6% F1)

### Key Concepts
- **Efficient Market Hypothesis**: Why reversals shouldn't exist
- **Technical Analysis**: Support, resistance, peaks, troughs
- **Class Imbalance**: Handling rare events in ML
- **Feature Engineering**: From indicators to predictive features
- **Time Series Cross-Validation**: Preventing data leakage

---

## 🎓 Author

**Shaik Abdus Sattar**
- GitHub: [@Seventie](https://github.com/Seventie)
- Location: Coimbatore, Tamil Nadu, India
- University: Amrita Vishwa Vidyapeetham

---

## 📝 License

This project is provided for educational and competition purposes.

---

**Last Updated**: December 18, 2025

*For questions or suggestions, open an issue or reach out!*
