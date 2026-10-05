7# Predicting Movie Rental Durations & Feature Significance Analysis

## Overview
This repository contains a predictive modeling and statistical testing framework developed for a DVD rental firm[span_0](start_span)[span_0](end_span). The goal is to help the rental inventory team predict how many days a customer will keep a rented movie (`rental_length_days`). Additionally, hypothesis testing via Ordinary Least Squares (OLS) regression is performed to determine which rental features (e.g., rental rate, replacement cost, bonus features) significantly influence rental duration.

---

## Hypothesis Testing Framework

To determine if specific rental attributes significantly affect duration, we test the regression coefficients ($\beta_i$) using $t$-tests:

*   **Null Hypothesis ($H_0$):** $\beta_i = 0$ (The feature has no statistically significant effect on rental duration).
*   **Alternative Hypothesis ($H_1$):** $\beta_i \neq 0$ (The feature significantly affects rental duration).

**Decision Criteria:** 
A feature is considered statistically significant if its $p$-value $< 0.05$.

---

## Workflow & Pipeline

1. **Feature Engineering:**
   * Calculate target variable `rental_length_days` from timestamps.
   * Parse categorical text features (`special_features`) into binary indicators like `deleted_scenes` and `behind_the_scenes`.
2. **Statistical Inference (OLS Hypothesis Testing):**
   * Fit an Ordinary Least Squares (OLS) model to evaluate $p$-values and $t$-statistics for each feature parameter.
3. **Feature Selection (Lasso Regularization):**
   * Apply Lasso regression ($\text{L1 penalty}$) to prune non-essential or redundant features.
4. **Predictive Modeling & Evaluation:**
   * Train and compare Linear Regression and Random Forest Regressor models using selected features.
   * Evaluate model performance using Mean Squared Error ($\text{MSE}$).

---

## Installation & Execution

### Prerequisites
* Python 3.8 or higher

### Setup Instructions
1. Clone this repository:
   ```bash
   git clone [https://github.com/your-username/movie-rental-duration-prediction.git](https://github.com/your-username/movie-rental-duration-prediction.git)
   cd movie-rental-duration-prediction

