#!/usr/bin/env python
import argparse
import sys
from rich.console import Console
from rich.panel import Panel

# Console for beautiful printing
console = Console()

def print_banner(text):
    console.print(Panel(f"[bold green]{text}[/bold green]", expand=False, border_style="green"))

def main():
    parser = argparse.ArgumentParser(
        description="🏆 WM Predictor ML Pipeline Orchestrator 🏆",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Pipeline step to execute")
    
    # 1. Scrape step
    scrape_parser = subparsers.add_parser("scrape", help="Collect raw data from Transfermarkt & Sofascore")
    scrape_parser.add_argument("--step", choices=["leagues", "tournaments", "ucl", "squads", "historical", "all"], default="all",
                               help="Specific scraping step to run")
    
    # 2. Features step
    features_parser = subparsers.add_parser("features", help="Process raw data & engineer features and datasets")
    features_parser.add_argument("--step", choices=["engineering", "dataset", "all"], default="all",
                                 help="Specific features step to run")
    
    # 3. Train step
    train_parser = subparsers.add_parser("train", help="Optimize & train the XGBoost match outcomes model")
    train_parser.add_argument("--tune", action="store_true", help="Perform grid search hyperparameter tuning")
    train_parser.add_argument("--model-name", default="xgboost_wm_modelV4.joblib", help="Filename of trained model to save")
    
    # 4. Info step (model inspection)
    info_parser = subparsers.add_parser("info", help="Inspect and display accuracy/stats of a trained model by its file name")
    info_parser.add_argument("--model-name", default="xgboost_wm_modelV4.joblib", help="Filename of the saved model in models/ folder")
    
    # 5. Compare step (model leaderboard)
    subparsers.add_parser("compare", help="Compare accuracies and stats of all saved models in models/ folder side-by-side")
    
    # 6. Simulate Groups step
    sim_groups_parser = subparsers.add_parser("simulate-groups", help="Run a single simulation of the World Cup Group Stage")
    sim_groups_parser.add_argument("--model-name", default="xgboost_wm_modelV4.joblib", help="Filename of the trained model to use")
    
    # 7. Simulate Tournament step
    sim_tournament_parser = subparsers.add_parser("simulate-tournament", help="Run Monte Carlo simulations of the complete World Cup")
    sim_tournament_parser.add_argument("--sims", type=int, default=1000, help="Number of Monte Carlo simulations to run")
    sim_tournament_parser.add_argument("--model-name", default="xgboost_wm_modelV4.joblib", help="Filename of the trained model to use")
    
    # 8. End-to-end pipeline step
    all_parser = subparsers.add_parser("all", help="Run the full pipeline end-to-end (data -> features -> train -> simulation)")
    all_parser.add_argument("--tune", action="store_true", help="Perform grid search during training phase")
    all_parser.add_argument("--model-name", default="xgboost_wm_modelV4.joblib", help="Filename of the model to save and use")
    all_parser.add_argument("--sims", type=int, default=1000, help="Number of simulations to run")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
        
    try:
        # Import pipeline functions lazily to avoid unnecessary module loading when displaying help
        if args.command == "scrape" or args.command == "all":
            from src.data.scraping import (
                scrape_all_leagues, scrape_historical_tournaments,
                scrape_ucl_data, scrape_all_squads, fetch_historical_matches
            )
            
            if args.command == "scrape":
                print_banner("RUNNING SCRAPING PIPELINE")
                if args.step in ["leagues", "all"]:
                    scrape_all_leagues()
                if args.step in ["tournaments", "all"]:
                    scrape_historical_tournaments()
                if args.step in ["ucl", "all"]:
                    scrape_ucl_data()
                if args.step in ["squads", "all"]:
                    scrape_all_squads()
                if args.step in ["historical", "all"]:
                    fetch_historical_matches()
                console.print("[bold green]✓ Scraping step finished successfully![/bold green]")
                
        if args.command == "features" or args.command == "all":
            from src.features.engineering import (
                clean_market_values, calculate_club_chemistry, calculate_weighted_ratings,
                calculate_ucl_experience, calculate_tournament_experience, scrape_fifa_ranking
            )
            from src.features.dataset import (
                build_master_dataset, create_matchups, build_training_data
            )
            from src.features.elo_calculator import calculate_elo_ratings
            
            print_banner("RUNNING FEATURE ENGINEERING & DATASET GENERATION")
            
            if args.command == "all" or args.step in ["engineering", "all"]:
                # 1. Fetch raw rankings/stats
                scrape_fifa_ranking()
                clean_market_values()
                calculate_club_chemistry()
                calculate_weighted_ratings()
                calculate_ucl_experience()
                calculate_tournament_experience()
                
                # 2. Dynamic zero-sum ELO ratings calculation (stabilized chronologically since 2010)
                calculate_elo_ratings()
                
            if args.command == "all" or args.step in ["dataset", "all"]:
                # 3. Build training sets and matchup matrices incorporating ELO Deltas
                build_master_dataset()
                create_matchups()
                build_training_data()
                
            console.print("[bold green]✓ Feature and dataset engineering finished successfully![/bold green]")
            
        if args.command == "train" or args.command == "all":
            from src.models.train import train_model
            
            print_banner("RUNNING MODEL TRAINING")
            model_name = getattr(args, "model_name", "xgboost_wm_modelV4.joblib")
            tune = getattr(args, "tune", False)
            train_model(model_name=model_name, use_tuning=tune)
            console.print("[bold green]✓ Model training finished successfully![/bold green]")
            
        if args.command == "info":
            from src.models.train import evaluate_saved_model
            
            print_banner("RUNNING MODEL EVALUATION INSPECT")
            model_name = getattr(args, "model_name", "xgboost_wm_modelV4.joblib")
            evaluate_saved_model(model_name=model_name)
            
        if args.command == "compare":
            from src.models.train import compare_models
            
            print_banner("RUNNING MULTI-MODEL COMPARISON LEADERBOARD")
            compare_models()
            
        if args.command == "simulate-groups":
            from src.simulation.group_stage import simulate_group_stage
            
            print_banner("RUNNING GROUP STAGE SIMULATION")
            model_name = getattr(args, "model_name", "xgboost_wm_modelV4.joblib")
            simulate_group_stage(model_name=model_name)
            console.print("[bold green]✓ Group stage simulation complete![/bold green]")
            
        if args.command == "simulate-tournament" or args.command == "all":
            from src.simulation.tournament import simulate_tournament
            
            print_banner("RUNNING TOURNAMENT MONTE CARLO SIMULATIONS")
            sims = getattr(args, "sims", 1000)
            model_name = getattr(args, "model_name", "xgboost_wm_modelV4.joblib")
            simulate_tournament(model_name=model_name, num_simulations=sims)
            console.print("[bold green]✓ Tournament simulation complete![/bold green]")
            
    except KeyboardInterrupt:
        console.print("\n[bold red]Pipeline execution interrupted by user.[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Pipeline failed with error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
