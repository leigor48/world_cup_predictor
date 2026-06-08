import os
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
from rich.console import Console
from rich.table import Table

console = Console()

def train_model(model_name="xgboost_wm_modelV4.joblib", use_tuning=False):
    """Trains an XGBoost multiclass classifier to predict match outcomes with time-decay sample weighting and smart class-weights."""
    data_path = os.path.join('data', 'processed', 'model_input', 'training_data.csv')
    if not os.path.exists(data_path):
        console.print(f"[bold red]Error: Training dataset not found at {data_path}. Run dataset generation first.[/bold red]")
        return
        
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    console.print(f"[bold blue]Loaded {len(df)} historical match records.[/bold blue]")
    
    # ---------------------------------------------------------
    # Time Decay (dynamic weights favoring recency)
    # ---------------------------------------------------------
    latest_date = df['date'].max()
    df['days_ago'] = (latest_date - df['date']).dt.days
    df['match_weight'] = np.exp(-df['days_ago'] / 1000)
    
    # FINETUNING: Removed non-significant Delta_Average_Age, added Delta_Top5_Density
    features = [
        'Delta_Total_Market_Value', 'Delta_Median_Top11_Value', 'Delta_Chemistry',
        'Delta_Form_Rating', 'Delta_UCL_Minutes', 'Delta_Tournament_Minutes',
        'Delta_TM_Value_Rank', 'Delta_Top5_Density', 'Delta_Elo',
        'Is_Neutral' 
    ]
    
    X = df[features]
    y = df['target']
    weights = df['match_weight']
    
    # Train/Test Split (stratify on y is a solid ML practice)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    
    # FINETUNING: Draw Balance (Increase sample weights of draws (class 1) to solve the draw recall problem)
    # Since draws make up roughly 24% of outcomes, scaling up their weights forces the trees to learn draw boundaries.
    draw_multiplier = 1.35
    w_train_adjusted = w_train.copy()
    w_train_adjusted[y_train == 1] = w_train_adjusted[y_train == 1] * draw_multiplier
    
    console.print(f"Split sizes -> Train: {len(X_train)} | Test: {len(X_test)}")
    
    if use_tuning:
        console.print("[bold yellow]Running Fine-Tuned Hyperparameter Optimization via GridSearchCV...[/bold yellow]")
        # FINETUNING: Expanded search grid with colsample_bytree and regularization
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.03, 0.05],
            'n_estimators': [50, 100, 150],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'reg_alpha': [0.0, 0.1, 0.5],
            'reg_lambda': [1.0, 1.5, 2.0]
        }
        
        xgb_base = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        grid_search = GridSearchCV(
            estimator=xgb_base,
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit GridSearchCV with training weights passed to fit
        grid_search.fit(X_train, y_train, sample_weight=w_train_adjusted)
        console.print("[bold green]Tuning completed successfully.[/bold green]")
        console.print(f"Best Parameters: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_
    else:
        # FINETUNING: Fine-tuned optimal default hyperparameters with colsample restriction
        console.print("[bold cyan]Training model with optimized fine-tuned hyperparameters...[/bold cyan]")
        best_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            learning_rate=0.03,
            max_depth=3,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=42,
            eval_metric='mlogloss'
        )
        best_model.fit(X_train, y_train, sample_weight=w_train_adjusted)
        
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    console.print(f"\n[bold green]Model Accuracy on Test Set: {accuracy * 100:.2f}%[/bold green]")
    console.print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Away Win (0)", "Draw (1)", "Home Win (2)"]))
    
    # Export Model + Metadata
    models_dir = os.path.join('models')
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, model_name)

    model_package = {
        "model": best_model,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": float(accuracy),
        "features": features,
        "training_rows": len(df),
        "draw_multiplier": draw_multiplier,
        "time_decay_days": 1000,
        "hyperparameters": best_model.get_params(),
        "model_version": model_name
    }

    joblib.dump(model_package, out_path)
    console.print(f"[bold green]✓ Model saved successfully to '{out_path}'![/bold green]")


def evaluate_saved_model(model_name="xgboost_wm_modelV4.joblib"):
    """Loads a saved XGBoost model and prints its accuracy, hyperparameters, and feature importances."""
    model_path = os.path.join('models', model_name)
    data_path = os.path.join('data', 'processed', 'model_input', 'training_data.csv')
    
    if not os.path.exists(model_path):
        console.print(f"[bold red]Error: Model file '{model_name}' not found in the models/ directory.[/bold red]")
        return
    if not os.path.exists(data_path):
        console.print(f"[bold red]Error: Training dataset not found at {data_path}.[/bold red]")
        return
        
    # Load Model
    model = joblib.load(model_path)
    
    # Load Data and replicate Train/Test Split
    df = pd.read_csv(data_path)
    features = [
        'Delta_Total_Market_Value', 'Delta_Median_Top11_Value', 'Delta_Chemistry',
        'Delta_Form_Rating', 'Delta_UCL_Minutes', 'Delta_Tournament_Minutes',
        'Delta_TM_Value_Rank', 'Delta_Top5_Density', 'Delta_Elo',
        'Is_Neutral' 
    ]
    
    X = df[features]
    y = df['target']
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Render inspect tables
    console.print(f"\n[bold green]📊 MODEL STATISTICS: {model_name}[/bold green]")
    console.print("="*60)
    
    # 1. Hyperparameters table
    params_table = Table(title="Model Hyperparameters", show_header=True, header_style="bold magenta")
    params_table.add_column("Hyperparameter", style="cyan")
    params_table.add_column("Value", style="green")
    
    try:
        params = model.get_params()
        for k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'eval_metric', 'objective']:
            if k in params:
                params_table.add_row(k, str(params[k]))
    except Exception:
        # Generic parameter fallback
        params_table.add_row("Details", "Non-standard model properties")
        
    console.print(params_table)
    
    # 2. Accuracy Summary
    console.print(f"\n[bold]Model Performance on Test Set:[/bold]")
    console.print(f"  -> [bold green]Accuracy (Genauigkeit): {accuracy * 100:.2f}%[/bold green]")
    console.print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Away Win (0)", "Draw (1)", "Home Win (2)"]))
    
    # 3. Feature Importance Table
    try:
        importances = model.feature_importances_
        imp_table = Table(title="Relative Feature Importances", show_header=True, header_style="bold yellow")
        imp_table.add_column("Feature", style="cyan")
        imp_table.add_column("Importance Weight", justify="right", style="green")
        
        sorted_indices = np.argsort(importances)[::-1]
        for idx in sorted_indices:
            imp_table.add_row(features[idx], f"{importances[idx] * 100:.2f}%")
            
        console.print(imp_table)
    except Exception:
        pass


def compare_models():
    """Loops through all files in the models/ directory, evaluates each on the test set, and prints a comparison table."""
    models_dir = 'models'
    data_path = os.path.join('data', 'processed', 'model_input', 'training_data.csv')
    
    if not os.path.exists(models_dir):
        console.print("[bold red]Error: 'models/' directory does not exist.[/bold red]")
        return
    if not os.path.exists(data_path):
        console.print("[bold red]Error: Training dataset not found.[/bold red]")
        return
        
    # Get all potential model files
    files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') or f.endswith('.model')]
    if not files:
        console.print("[bold yellow]No models found in the models/ folder.[/bold yellow]")
        return
        
    # Replicate train/test split
    df = pd.read_csv(data_path)
    features = [
        'Delta_Total_Market_Value', 'Delta_Median_Top11_Value', 'Delta_Chemistry',
        'Delta_Form_Rating', 'Delta_UCL_Minutes', 'Delta_Tournament_Minutes',
        'Delta_TM_Value_Rank', 'Delta_Top5_Density', 'Delta_Elo', 
        'Is_Neutral' 
    ]
    X = df[features]
    y = df['target']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Rich Table setup
    table = Table(title="🤖 Model Comparison Leaderboard 🤖", show_header=True, header_style="bold magenta")
    table.add_column("Model File", style="cyan")
    table.add_column("Accuracy (Genauigkeit)", justify="right", style="green")
    table.add_column("Hyperparameters (Summary)", style="yellow")
    table.add_column("Status / Note", style="white")
    
    results = []
    
    for filename in sorted(files):
        model_path = os.path.join(models_dir, filename)
        try:
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Hyperparameter summary
            param_summary = "N/A"
            try:
                params = model.get_params()
                lr = params.get('learning_rate', '?')
                depth = params.get('max_depth', '?')
                n_est = params.get('n_estimators', '?')
                param_summary = f"lr={lr}, depth={depth}, n_est={n_est}"
            except Exception:
                pass
                
            results.append({
                'filename': filename,
                'accuracy': accuracy,
                'params': param_summary,
                'status': "✓ Active & Functional"
            })
        except Exception as e:
                    error_msg = str(e)
                    console.print(f"[bold red]DEBUG ERROR for {filename}:[/bold red] {error_msg}")
                    
                    results.append({
                        'filename': filename,
                        'accuracy': -1.0,
                        'params': "N/A",
                        'status': f"❌ {error_msg[:60]}" if len(error_msg) > 60 else f"❌ {error_msg}"
                    })
            
    # Sort results by accuracy descending, placing failed models at the bottom
    results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    for res in results:
        acc_str = f"{res['accuracy'] * 100:.2f}%" if res['accuracy'] >= 0 else "N/A"
        status_color = "green" if "✓" in res['status'] else "red"
        table.add_row(
            res['filename'],
            acc_str,
            res['params'],
            f"[{status_color}]{res['status']}[/{status_color}]"
        )
        
    console.print(table)
