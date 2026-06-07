import os
import itertools
import pandas as pd
import numpy as np
import joblib

def simulate_group_stage(model_name="xgboost_wm_modelV4.joblib"):
    """Runs a single simulation of the official 12 groups of the World Cup using the specified model."""
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
        'Delta_Average_Age', 'Delta_TM_Value_Rank', 'Delta_FIFA_Rank', 'Delta_FIFA_Points',
        'Is_Neutral'
    ]
    
    print(f"🏆 STARTING GROUP SIMULATION USING MODEL: {model_name} 🏆")
    print("="*60)
    
    for group_name, teams in groups.items():
        group_standings = {team: 0 for team in teams}
        print(f"\n--- GROUP {group_name} ---")
        
        matchups = list(itertools.combinations(teams, 2))
        
        for team_a, team_b in matchups:
            data_a = master_df[master_df['Country'].str.lower() == team_a.lower()]
            data_b = master_df[master_df['Country'].str.lower() == team_b.lower()]
            
            if data_a.empty or data_b.empty:
                print(f"⚠️ Warning: {team_a} or {team_b} not found in the dataset.")
                continue
                
            data_a = data_a.iloc[0]
            data_b = data_b.iloc[0]
            
            delta_dict = {
                'Delta_Total_Market_Value': data_a['Total_Market_Value_mEUR'] - data_b['Total_Market_Value_mEUR'],
                'Delta_Median_Top11_Value': data_a['Median_Top11_Market_Value_mEUR'] - data_b['Median_Top11_Market_Value_mEUR'],
                'Delta_Chemistry': data_a['Chemistry_Score'] - data_b['Chemistry_Score'],
                'Delta_Form_Rating': data_a['Current_Form_Rating'] - data_b['Current_Form_Rating'],
                'Delta_UCL_Minutes': data_a['Total_UCL_Minutes'] - data_b['Total_UCL_Minutes'],
                'Delta_Tournament_Minutes': data_a['Total_Tournament_Minutes'] - data_b['Total_Tournament_Minutes'],
                'Delta_Average_Age': data_a['Average_Age'] - data_b['Average_Age'],
                'Delta_TM_Value_Rank': data_a['TM_Value_Rank'] - data_b['TM_Value_Rank'],
                'Delta_FIFA_Rank': data_a['FIFA_Rank'] - data_b['FIFA_Rank'],
                'Delta_FIFA_Points': data_a['FIFA_Points'] - data_b['FIFA_Points'],
                'Is_Neutral': 1
            }
            
            X_pred = pd.DataFrame([delta_dict])[features]
            probs = model.predict_proba(X_pred)[0]
            outcome = np.random.choice([0, 1, 2], p=probs)
            
            if outcome == 2:
                print(f"  {team_a} beats {team_b} (Win Prob: {probs[2]*100:.1f}%)")
                group_standings[team_a] += 3
            elif outcome == 0:
                print(f"  {team_b} beats {team_a} (Win Prob: {probs[0]*100:.1f}%)")
                group_standings[team_b] += 3
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
