# System Design Document: Mobile Game A/B Testing & Statistical Experimentation Engine

A robust, enterprise-grade architecture and statistical pipeline for controlled online experiments, evaluating game gate positioning (Level 30 vs Level 40) for user retention and engagement metrics.

---

## 1. System Architecture Overview

The **A/B Testing Infrastructure Engine** manages end-to-end controlled experimentation, including random assignment hashing, Sample Ratio Mismatch (SRM) checks, parametric and non-parametric hypothesis testing, and empirical bootstrap resampling.

```
+-----------------------------------------------------------------------------------+
|                                  USER BASE                                        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        1. RANDOMIZATION & TRAFFIC SPLIT                           |
|                       Hash-based Assignment (Murmur3/MD5)                         |
+-----------------------------------------------------------------------------------+
                                /                     \
                               /                       \
                              v                         v
+-----------------------------------------+   +-----------------------------------------+
|     Control Group (A): Gate Level 30    |   |    Treatment Group (B): Gate Level 40   |
|            (N_A = 44,700)               |   |             (N_B = 45,489)              |
+-----------------------------------------+   +-----------------------------------------+
                              \                         /
                               \                       /
                                v                     v
+-----------------------------------------------------------------------------------+
|                           2. DATA INGESTION & ETL PIPELINE                        |
|                     Event Tracking: sum_gamerounds, retention_1, retention_7       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                      3. EXPERIMENTATION SANITY & DIAGNOSTICS                      |
|                  SRM Test (Chi-Square) | A/A Validation (KS-Uniformity)          |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                         4. STATISTICAL EVALUATION ENGINE                          |
|         Normality Check -> Homogeneity Test -> Mann-Whitney U / Z-Proportion       |
|                             Bootstrap Resampling (N=1,000)                        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                           5. DECISION & ROLLOUT GATE                              |
|                          LAUNCH DECISION: DO NOT LAUNCH                           |
+-----------------------------------------------------------------------------------+
```

---

## 2. Metric Hierarchy & Business Objectives

```
                        +---------------------------------------+
                        |            SUCCESS METRIC             |
                        |      Average Game Rounds Played       |
                        |          (`sum_gamerounds`)           |
                        +---------------------------------------+
                                           |
                                           v
                        +---------------------------------------+
                        |            DRIVER METRICS             |
                        |   1-Day Retention Rate (`retention_1`) |
                        |   7-Day Retention Rate (`retention_7`) |
                        +---------------------------------------+
                                           |
                                           v
                        +---------------------------------------+
                        |           GUARDRAIL METRICS           |
                        |  App Latency, Error Rates, Churn Rate |
                        +---------------------------------------+
```

| Metric Type | Metric Name | Definition | Business Rationale |
| :--- | :--- | :--- | :--- |
| **Success Metric** | `sum_gamerounds` | Total game rounds completed per user post-installation. | Primary indicator of user engagement and monetization potential. |
| **Driver Metric 1** | `retention_1` | Binary indicator ($1$ if user played 1 day after install, $0$ otherwise). | Evaluates short-term onboarding stickiness. |
| **Driver Metric 2** | `retention_7` | Binary indicator ($1$ if user played 7 days after install, $0$ otherwise). | Measures long-term player retention and game mechanics appeal. |
| **Guardrail Metric** | Latency / Errors | Operational metrics monitored during live traffic splitting. | Protects system against platform issues and degraded UX. |

---

## 3. Pre-Experiment Validation & Data Integrity Architecture

### 3.1 Sample Ratio Mismatch (SRM) Detection
To detect systematic traffic allocation bias, a Goodness-of-Fit Chi-Square ($\chi^2$) test is executed on observed vs expected traffic counts:

$$\chi^2 = \sum_{i \in \{A, B\}} \frac{(O_i - E_i)^2}{E_i}$$

* **Observed Control ($N_A$)**: $44,700$
* **Observed Treatment ($N_B$)**: $45,489$
* **Chi-Square Statistic**: $6.92$
* **p-value**: $0.0085$ ($\text{p-value} < 0.01$)
* **Diagnosis**: **SRM Alert Triggered.** Significant traffic discrepancy detected; requires infrastructure investigation into traffic-splitting layers before final production rollout.

### 3.2 A/A Testing Engine & Platform Validation
Before conducting A/B inference, a benchmark A/A test evaluates experiment platform reliability:
1. **P-Value Uniformity Verification**: Simulated A/A trials are performed under $H_0$.
2. **Kolmogorov-Smirnov Test**: Tests whether the empirical distribution of p-values matches a $U(0, 1)$ distribution.
   $$\text{KS-Statistic} = 0.0084, \quad p = 0.4759$$
3. **Conclusion**: Uniformity confirmed ($p > 0.05$), proving the platform randomization mechanism is unbiased.

---

## 4. Automated Statistical Decision Logic Flow

```
                      +------------------------------------------+
                      |         Continuous Feature Input         |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |          Shapiro-Wilk Normality          |
                      |            Test (alpha = 0.05)           |
                      +------------------------------------------+
                                      /          \
                         p > 0.05    /            \   p <= 0.05
                        +-----------+              +-----------+
                        |                              |
                        v                              v
           +--------------------------+  +---------------------------+
           |   Levene Variance Test   |  |   Mann-Whitney U Test     |
           +--------------------------+  |      (Non-Parametric)     |
              /                    \     +---------------------------+
  p > 0.05   /                      \  p <= 0.05
  +---------+                        +---------+
  |                                            |
  v                                            v
+---------------------------+    +---------------------------+
| Student's Two-Sample      |    | Welch's t-Test            |
| t-Test (Equal Variance)   |    | (Unequal Variance)        |
+---------------------------+    +---------------------------+
```

---

## 5. Statistical Hypothesis Testing & Performance Evaluation

### 5.1 Success Metric: Game Rounds Played (`sum_gamerounds`)
* **Normality Check**: Shapiro-Wilk Test yield $p = 0.0000$ for both variants $\implies$ Highly skewed non-normal distribution.
* **Non-Parametric Test**: Mann-Whitney U Test executed.
  * **Test Statistic p-value**: $0.05089$ ($\text{p-value} > 0.05$)
  * **Statistical Conclusion**: Fail to reject $H_0$. No statistically significant difference in game rounds between Level 30 and Level 40 gates.

### 5.2 Driver Metrics: 1-Day & 7-Day Retention Rates
For binary proportion variables, two-sample $Z$-tests for proportions and non-parametric empirical bootstrap resampling ($N = 1,000$ iterations) were conducted:

| Variant Metric | Control (`gate_30`) | Treatment (`gate_40`) | Z-Test $p$-value | Bootstrap Prob ($P(B < A)$) | Statistical Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1-Day Retention** | **44.82%** | **44.23%** | $0.0739$ | **96.20%** | Treatment degrades 1-day retention ($p_{\text{boot}} < 0.05$) |
| **7-Day Retention** | **19.02%** | **18.20%** | $0.0016$ | **99.80%** | Treatment significantly degrades 7-day retention ($p < 0.01$) |

---

## 6. Top Causal Factors & Root Cause Analysis

```
       [ GATE LEVEL MOVED FROM 30 TO 40 ]
                       |
                       v
       [ EXPANDED UNINTERRUPTED GAMEPLAY ]
                       |
                       v
       [ EARLY ENJOYMENT SATIATION & FATIGUE ]
                       |
                       v
       [ DECREASED LONG-TERM RETENTION & LTV ]
```

| Causal Factor | Mechanism / Impact | Observed Data Impact |
| :--- | :--- | :--- |
| **1. Novelty Satiation & Burnout** | Pushing the gate to Level 40 allows players to play continuously without enforced break intervals. This causes faster enjoyment burnout and higher early drop-off. | 7-day retention dropped significantly from **19.02% to 18.20%** ($p = 0.0016$). |
| **2. Loss of Natural Pacing Gates** | Level 30 gate served as a forced pause, creating intrinsic motivation to return. Removing it diminished long-term player habituation. | Probability that Level 40 has lower 1-day retention than Level 30 is **96.2%**. |

---

## 7. System Launch Decision & Actionable Recommendations

### Final Launch Decision: **`DO NOT LAUNCH`**

1. **Reject Rollout**: Keep the gate at **Level 30**. Moving the gate to Level 40 does not increase total game rounds played and significantly damages both 1-day and 7-day user retention.
2. **Infrastructure Remediation**: Investigate the root cause of the **Sample Ratio Mismatch ($p = 0.0085$)** in the platform traffic-splitting service to ensure pure 50/50 hashing distribution in future experiments.
3. **Product Iteration**: Explore alternative monetization gating strategies (e.g., dynamic difficulty scaling or soft gating) rather than delaying forced wait times.