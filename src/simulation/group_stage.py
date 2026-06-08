import os
import itertools
import pandas as pd
import numpy as np
import joblib

def simulate_group_stage(model_name="xgboost_wm_modelV4.joblib"):
    """Runs a single simulation of the official 12 groups of the World Cup using smart co-host home advantage."""
    model_path = os.path.join('models', model_name)
    master_path = os.path.join('data', 'processed', 'features', 'MASTER_dataset.csv')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_name}' not found in the models/ directory.")
        return
    if not os.path.exists(master_path):
        print("Error: MASTER_dataset.csv missing.")
        return
        
    model = joblib.load(model_path)
    master_df = pd.read_csv(master_path)
    
    groups = {
        'A': ['Mexico', 'South Africa', 'South Korea', 'Czech Republic'],
        'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
        'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
        'D': ['USA', 'Paraguay', 'Australia', 'Turkey'],
        'E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
        'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
        'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
        'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
        'I': ['France', 'Senegal', 'Iraq', 'Norway'],
        'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
        'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
        'L': ['England', 'Croatia', 'Ghana', 'Panama']
    }
    
    features = [
        'Delta_Total_Market_Value', 'Delta_Median_Top11_Value', 'Delta_Chemistry',
        'Delta_Form_Rating', 'Delta_UCL_Minutes', 'Delta_Tournament_Minutes',
        'Delta_TM_Value_Rank', 'Delta_FIFA_Rank', 'Delta_FIFA_Points', 'Delta_Top5_Density',
        'Delta_Elo',
        'Is_Neutral'
    ]
    
    # 2026 co-hosts
    hosts = ['usa', 'canada', 'mexico']
    
    print(f"🏆 STARTING GROUP SIMULATION USING MODEL: {model_name} 🏆")
    print("🌍 Smart Host Advantage: USA, Canada, and Mexico play as home teams (Is_Neutral=0)")
    print("="*75)
    
    for group_name, teams in groups.items():
        group_standings = {team: 0 for team in teams}
        print(f"\n--- GROUP {group_name} ---")
        
        matchups = list(itertools.combinations(teams, 2))
        
        for team_a, team_b in matchups:
            # 1. Determine Home/Away with Host Advantage logic
            a_is_host = team_a.lower() in hosts
            b_is_host = team_b.lower() in hosts
            
            if a_is_host and not b_is_host:
                home_team, away_team = team_a, team_b
                is_neutral = 0
                reversed_order = False
            elif b_is_host and not a_is_host:
                home_team, away_team = team_b, team_a
                is_neutral = 0
                reversed_order = True
            else:
                home_team, away_team = team_a, team_b
                is_neutral = 1
                reversed_order = False
                
            data_home = master_df[master_df['Country'].str.lower() == home_team.lower()]
            data_away = master_df[master_df['Country'].str.lower() == away_team.lower()]
            
            if data_home.empty or data_away.empty:
                print(f"⚠️ Warning: {home_team} or {away_team} not found in the dataset.")
                continue
                
            data_home = data_home.iloc[0]
            data_away = data_away.iloc[0]
            
            delta_dict = {
                'Delta_Total_Market_Value': data_home['Total_Market_Value_mEUR'] - data_away['Total_Market_Value_mEUR'],
                'Delta_Median_Top11_Value': data_home['Median_Top11_Market_Value_mEUR'] - data_away['Median_Top11_Market_Value_mEUR'],
                'Delta_Chemistry': data_home['Chemistry_Score'] - data_away['Chemistry_Score'],
                'Delta_Form_Rating': data_home['Current_Form_Rating'] - data_away['Current_Form_Rating'],
                'Delta_UCL_Minutes': data_home['Total_UCL_Minutes'] - data_away['Total_UCL_Minutes'],
                'Delta_Tournament_Minutes': data_home['Total_Tournament_Minutes'] - data_away['Total_Tournament_Minutes'],
                'Delta_TM_Value_Rank': data_home['TM_Value_Rank'] - data_away['TM_Value_Rank'],
                'Delta_FIFA_Rank': data_home['FIFA_Rank'] - data_away['FIFA_Rank'],
                'Delta_FIFA_Points': data_home['FIFA_Points'] - data_away['FIFA_Points'],
                'Delta_Top5_Density': data_home['Top5_League_Density'] - data_away['Top5_League_Density'],
                'Delta_Elo': data_home['ELO_Rating'] - data_away['ELO_Rating'],
                'Is_Neutral': is_neutral
            }
            
            X_pred = pd.DataFrame([delta_dict])[features]
            probs = model.predict_proba(X_pred)[0]
            outcome = np.random.choice([0, 1, 2], p=probs)
            
            # Map predictions back to the original order (team_a vs team_b)
            if not reversed_order:
                if outcome == 2:
                    print(f"  {team_a} beats {team_b} (Win Prob: {probs[2]*100:.1f}%) [Home Advantage]")
                    group_standings[team_a] += 3
                elif outcome == 0:
                    print(f"  {team_b} beats {team_a} (Win Prob: {probs[0]*100:.1f}%)")
                    group_standings[team_b] += 3
                else:
                    print(f"  {team_a} and {team_b} draw (Draw Prob: {probs[1]*100:.1f}%)")
                    group_standings[team_a] += 1
                    group_standings[team_b] += 1
            else:
                # Reversed order -> probs[2] is team_b (home) win, probs[0] is team_a (away) win
                if outcome == 2:
                    print(f"  {team_b} beats {team_a} (Win Prob: {probs[2]*100:.1f}%) [Home Advantage]")
                    group_standings[team_b] += 3
                elif outcome == 0:
                    print(f"  {team_a} beats {team_b} (Win Prob: {probs[0]*100:.1f}%)")
                    group_standings[team_a] += 3
                else:
                    print(f"  {team_a} and {team_b} draw (Draw Prob: {probs[1]*100:.1f}%)")
                    group_standings[team_a] += 1
                    group_standings[team_b] += 1
                
        sorted_standings = sorted(
            group_standings.items(), 
            key=lambda x: (x[1], master_df[master_df['Country'].str.lower() == x[0].lower()].iloc[0]['FIFA_Points']), 
            reverse=True
        )
        
        print("\nStandings:")
        for rank, (team, points) in enumerate(sorted_standings, 1):
            print(f"  {rank}. {team:<20} {points} Pts.")
