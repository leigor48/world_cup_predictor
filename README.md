# ⚽ World Cup Predictor & Monte Carlo Tournament Simulator

An end-to-end, production-grade Machine Learning pipeline designed to predict international football match outcomes and simulate complete tournament brackets (from Group Stage to lifting the trophy) using **XGBoost** and **Monte Carlo Simulations**.

---

## 📋 Executive Summary
Predicting international football matches is a highly complex problem due to rapidly shifting squad dynamics, variable player form across domestic leagues, and scarce historical head-to-head national data. 

This repository solves these challenges with a professional, modular ML architecture:
1. **Dynamic Scraping Engine**: Collects player statistics from SofaScore and Transfermarkt across **17 worldwide leagues** and major international cups.
2. **Advanced Feature Engineering**: Combines squad market values, a custom-designed **Club Chemistry Index**, recency-weighted player form, and UEFA Champions League experience minutes.
3. **Recency-Weighted XGBoost Classifier**: Implements an exponential time-decay weighting function ($w = e^{-\Delta t / 1000}$) to favor recent match results.
4. **Monte Carlo Bracket Simulator**: Simulates the complete 48-team, 12-group World Cup tournament 10,000+ times to deliver reliable progression probabilities.

---

## 🛠️ Repository Architecture

```text
wm_predictor/
├── src/
│   ├── data/                 # Raw scraping & data collection modules (SofaScore & TM)
│   │   ├── scraping.py
│   ├── features/             # Feature extraction and dataset compiler modules
│   │   ├── engineering.py    # Club chemistry, form ratings, UCL & cup experience
│   │   └── dataset.py        # Pairs matchups and compiles master training CSVs
│   ├── models/               # Model architectures and training optimization
│   │   └── train.py          # Multiclass Recency-Weighted XGBoost model
│   └── simulation/           # Monte Carlo tournament bracket engines
│       ├── group_stage.py
│       └── tournament.py
├── notebooks/                # Presentation layer notebooks
│   ├── 01_model_training.ipynb           # Model training, hyperparameter tuning, evaluation
│   ├── 02_wm_oracle.ipynb                # Pairwise team outcome interactive predictor
│   └── 03_tournament_simulation.ipynb    # 10,000x Monte Carlo World Cup bracket simulator
├── run_pipeline.py           # Unified Master CLI Orchestrator
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started & CLI Usage

All tasks in this project are orchestrated through a unified Python Command Line Interface (`run_pipeline.py`), making it incredibly easy to rerun or update the pipeline.

### Setup Environment
```bash
# Activate your virtual environment and install dependencies
pip install -r requirements.txt
```

### 1. Scrape All Raw Data
Fetches the latest player ratings, league data, tournament matches, squad rosters, and historical matches:
```bash
python run_pipeline.py scrape
```

### 2. Compute Features & Build Datasets
Computes the features (Chemistry, Form, UCL Minutes) and outputs the final training and simulation matchup matrices:
```bash
python run_pipeline.py features
```

### 3. Train & Tune the XGBoost Model
Trains the model on recency-weighted historical data. You can specify a custom model name to save your work, and use `--tune` to perform an exhaustive hyperparameter search:
```bash
python run_pipeline.py train --model-name my_custom_model.joblib --tune
```

### 4. Inspect Model Statistics & Performance
Displays accuracy, classification reports, hyperparameter values, and a structured table of relative feature importances for any saved model:
```bash
python run_pipeline.py info --model-name my_custom_model.joblib
```

### 5. Compare All Models (Leaderboard)
Inspects and compares all models saved in the `models/` directory, outputting a sorted leaderboard based on test accuracy. If any model fails (e.g., due to different feature shapes, regression formatting, or corrupt data), the CLI catches the error cleanly, logs it, and continues with the next model:
```bash
python run_pipeline.py compare
```

### 6. Run Group Stage Simulation
Run a single simulation of the official 12 World Cup groups using your specified model:
```bash
python run_pipeline.py simulate-groups --model-name my_custom_model.joblib
```

### 6. Run Monte Carlo Tournament Simulation
Run a full 1,000-run Monte Carlo bracket prediction with your specified model:
```bash
python run_pipeline.py simulate-tournament --sims 1000 --model-name my_custom_model.joblib
```

### 7. Run the Full Pipeline End-to-End
Runs scraping, feature engineering, training, and tournament simulations consecutively with a single command:
```bash
python run_pipeline.py all --model-name my_custom_model.joblib --sims 1000
```

---

## 📊 Feature Engineering Highlights

- **Club Chemistry Index**: Quantifies the squad synergy by calculating combinations of links of players who play together in the same domestic clubs.
- **Weighted Form Rating**: Extracts player performance ratings across the top 17 domestic leagues, weighted by the respective league's competitive coefficient.
- **UCL & Tournament Experience**: Aggregates total career Champions League and World Cup play minutes to measure tactical experience under high pressure.

---