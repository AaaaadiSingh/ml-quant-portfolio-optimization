# ML-Based Portfolio Optimization for Risk-Aware Investment Decision Making

## What is this project about?

When we build a portfolio using the traditional Markowitz Mean-Variance Optimization (MVO) approach, we usually assume that the expected returns and risk estimates we give the optimizer are reasonably accurate. In reality, financial estimates are noisy. A small error in the expected return of an asset can sometimes have a surprisingly large effect on the final portfolio weights because the optimizer tends to put more emphasis on the numbers that look most attractive.

This project explores whether we can make the process more robust by treating those estimates as **uncertain predictions rather than facts**.

The idea is to build the system in three main layers:

* **Machine Learning** is used as the estimation engine. Instead of relying only on historical average returns and volatility, models such as Linear Regression, Random Forest, and XGBoost will use market features to estimate future quantities.
* **Monte Carlo simulation** is used to model the uncertainty around those estimates. Rather than giving the optimizer one fixed prediction and trusting it completely, we generate many plausible versions of that prediction and see how the optimal portfolio changes.
* **Walk-forward validation** is used to make the evaluation realistic. At each point in time, the model will only use information that would actually have been available then, retrain, build the portfolio, and evaluate it on future data.

The resulting portfolios will be compared with commonly used approaches such as Equal Weight, Global Minimum Variance, Classic Maximum Sharpe, and Risk Parity.

The main goal is not simply to find a portfolio with the highest historical return. It is to investigate whether **accounting for estimation uncertainty can lead to more stable and reliable portfolio decisions out of sample.**

---

## The Core Idea

The problem can be thought of like this:

```text
Traditional approach:

Historical Data
      ↓
Estimate Returns & Risk
      ↓
Give ONE estimate to optimizer
      ↓
Optimizer trusts the estimate
      ↓
Portfolio weights
```

If the estimate is wrong, the optimizer can make a very different decision from what we intended.

This project instead follows:

```text
Historical Data
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
Estimated Returns / Risk
      ↓
"How uncertain are these estimates?"
      ↓
Monte Carlo Scenarios
      ↓
Optimize each scenario
      ↓
Aggregate the resulting portfolios
      ↓
Walk-Forward Backtest
      ↓
Evaluate Out-of-Sample Performance
```

---

## Pipeline Overview

```text
┌──────────────────────────────────────────────┐
│              Raw Market Data                 │
│          OHLCV / NSE-BSE / APIs              │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│           Data Cleaning & Features           │
│                                              │
│ Returns • Volatility • Momentum • Indicators │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│              ML Estimation Layer             │
│                                              │
│ Linear Regression • Random Forest • XGBoost  │
└──────────────────────┬───────────────────────┘
                       ↓
          ┌─────────────────────────┐
          │ Expected Return / Risk  │
          │       Estimates         │
          └────────────┬────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│          Monte Carlo Resampling              │
│                                              │
│ Treat estimates as uncertain and generate    │
│ multiple plausible scenarios                 │
│                                              │
│ Planned: N = 500–1000 scenarios              │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│           Portfolio Optimization             │
│                                              │
│ Re-solve the portfolio for each scenario     │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│            Weight Aggregation                 │
│                                              │
│ Mean / Median of scenario portfolios         │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│            Walk-Forward Backtest             │
│                                              │
│ Retrain → Predict → Optimize → Test → Move   │
│ forward through time                         │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│             Performance Analysis             │
│                                              │
│ Sharpe • Sortino • CAGR • MDD • Calmar       │
│ VaR • CVaR                                   │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│              Benchmark Comparison            │
│                                              │
│ Equal Weight • Min Variance • Max Sharpe     │
│ Risk Parity                                  │
└──────────────────────────────────────────────┘
```

---

## Research Question

> **Can machine-learning-based estimation, combined with Monte Carlo modeling of estimation uncertainty, make portfolio optimization more robust when evaluated on unseen future data?**

The important part is the **out-of-sample** evaluation. A strategy that looks good only because it fits historical data is not enough.

---

## What I am trying to investigate

The project will gradually answer a few questions:

1. Can ML models estimate future returns or volatility better than simple historical estimates?
2. How sensitive are portfolio weights to errors in these estimates?
3. Does explicitly modeling estimation uncertainty change the resulting allocations?
4. Does repeatedly optimizing across Monte Carlo scenarios produce more stable weights?
5. How do these portfolios behave during periods of higher market volatility?
6. Do any observed improvements remain when everything is evaluated strictly out of sample?

These questions will be tested rather than assumed.

---

## Planned Models

### Machine Learning

The initial experiments will include:

* Linear Regression
* Random Forest
* XGBoost

These models will be treated primarily as **estimation tools**, not as black-box trading systems.

### Portfolio Benchmarks

The ML + Monte Carlo approach will be compared against:

* Equal Weight
* Global Minimum Variance
* Classic Maximum Sharpe
* Risk Parity

This gives the project simple and established reference points instead of evaluating the proposed method in isolation.

---

## Evaluation

The final evaluation will focus on both returns and risk.

Planned metrics include:

* Sharpe Ratio
* Sortino Ratio
* CAGR
* Maximum Drawdown
* Calmar Ratio
* Value at Risk (VaR)
* Conditional Value at Risk (CVaR)

The backtest will use a **walk-forward setup**, where future information is never intentionally made available to the model during training.

---

## Project Status

### Week 1 — Planning Phase

This repository currently contains the project structure, research documentation, and initial environment setup.

There is **no actual feature-engineering, ML, portfolio-optimization, or backtesting pipeline yet**. Those will be added incrementally as the project progresses.

The purpose of starting with documentation and structure is to make the development process reproducible and to keep track of how the research idea develops over the 16-week project.

---

## Repository Structure

```text
ml-quant-portfolio-optimization/
│
├── data/
│   ├── raw/                  # Raw market data
│   └── processed/            # Cleaned/processed datasets
│
├── docs/
│   └── synopsis.pdf          # Submitted project synopsis
│
├── notebooks/                # Exploratory analysis
│
├── src/                      # Main project code
│
├── tests/                    # Tests
│
├── results/
│   └── figures/              # Generated research figures
│
├── README.md
├── CONTEXT.md                # Detailed research roadmap
├── requirements.txt
├── CHANGELOG.md
├── .gitignore
└── LICENSE
```

---

## Documentation

**[Project Synopsis](docs/synopsis.pdf)**
The official academic synopsis submitted for the project.

**[Project Context & Roadmap](CONTEXT.md)**
The detailed project plan containing the architecture, mathematical formulation, 16-week roadmap, research considerations, and implementation notes.

---

## A Note on the Project

This is being developed as a **16-week minor academic project in quantitative finance**.

The aim is to build the system step by step, document the decisions along the way, and let the experiments determine what actually works rather than assuming that a more complicated model will automatically produce a better portfolio.

---

## Author

**Adi Aditya Singh**
12419051723, IIOT-B2

**Supervisor:** Dr. Ruchika Sehgal

University School of Automation and Robotics, GGSIPU

*Minor Academic Project - Quantitative Finance Research Pipeline*
