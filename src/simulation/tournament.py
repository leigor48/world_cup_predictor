import os
import itertools
import pandas as pd
import numpy as np
import joblib
from collections import defaultdict

def simulate_tournament(model_name="xgboost_wm_modelV4.joblib", num_simulations=1000):
    """Simulates multiple complete World Cup tournaments (Group Stage + Bracket) using the specified model."""
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
    
    all_teams = [team for group_teams in groups.values() for team in group_teams]
    
    print(f"Pre-calculating pairwise probabilities for all {len(all_teams) * (len(all_teams) - 1) // 2} matchups using {model_name}...")
    match_probs = {}
    team_fifa_points = {}
    
    for team in all_teams:
        team_data = master_df[master_df['Country'].str.lower() == team.lower()]
        team_fifa_points[team] = team_data.iloc[0]['FIFA_Points']
        
    for team_a, team_b in itertools.combinations(all_teams, 2):
        data_a = master_df[master_df['Country'].str.lower() == team_a.lower()].iloc[0]
        data_b = master_df[master_df['Country'].str.lower() == team_b.lower()].iloc[0]
        
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
        match_probs[(team_a, team_b)] = probs

    def play_ko_match(team_a, team_b):
        if (team_a, team_b) in match_probs:
            probs = match_probs[(team_a, team_b)]
            p_a = probs[2] / (probs[2] + probs[0])
            return team_a if np.random.rand() < p_a else team_b
        else:
            probs = match_probs[(team_b, team_a)]
            p_b = probs[2] / (probs[2] + probs[0])
            return team_b if np.random.rand() < p_b else team_a

    def play_round(teams):
        return [play_ko_match(teams[i], teams[i+1]) for i in range(0, len(teams), 2)]

    print(f"Simulating {num_simulations} World Cups...")
    ko_stats = defaultdict(lambda: {'R32': 0, 'R16': 0, 'QF': 0, 'SF': 0, 'Final': 0, 'Win': 0})
    
    for _ in range(num_simulations):
        winners, runners_up, thirds = {}, {}, []
        
        # 1. Group Stage
        for group_name, teams in groups.items():
            standings = {team: 0 for team in teams}
            for team_a, team_b in itertools.combinations(teams, 2):
                probs = match_probs[(team_a, team_b)]
                outcome = np.random.choice([0, 1, 2], p=probs)
                if outcome == 2: standings[team_a] += 3
                elif outcome == 0: standings[team_b] += 3
                else:
                    standings[team_a] += 1
                    standings[team_b] += 1
                    
            sorted_group = sorted(standings.items(), key=lambda x: (x[1], team_fifa_points[x[0]]), reverse=True)
            t1, t2, t3, _ = [t[0] for t in sorted_group]
            
            winners[group_name] = t1
            runners_up[group_name] = t2
            thirds.append((t3, sorted_group[2][1], team_fifa_points[t3]))
            
        best_thirds = [t[0] for t in sorted(thirds, key=lambda x: (x[1], x[2]), reverse=True)[:8]]
        
        # 2. Round of 32
        r32_teams = [
            winners['A'], best_thirds[0], winners['B'], best_thirds[1],
            winners['C'], best_thirds[2], winners['D'], best_thirds[3],
            winners['E'], best_thirds[4], winners['F'], best_thirds[5],
            winners['G'], best_thirds[6], winners['H'], best_thirds[7],
            winners['I'], runners_up['A'], winners['J'], runners_up['B'],
            winners['K'], runners_up['C'], winners['L'], runners_up['D'],
            runners_up['E'], runners_up['F'], runners_up['G'], runners_up['H'],
            runners_up['I'], runners_up['J'], runners_up['K'], runners_up['L']
        ]
        for t in r32_teams: ko_stats[t]['R32'] += 1
            
        # 3. Bracket Play
        r16_teams = play_round(r32_teams)
        for t in r16_teams: ko_stats[t]['R16'] += 1
            
        qf_teams = play_round(r16_teams)
        for t in qf_teams: ko_stats[t]['QF'] += 1
            
        sf_teams = play_round(qf_teams)
        for t in sf_teams: ko_stats[t]['SF'] += 1
            
        final_teams = play_round(sf_teams)
        for t in final_teams: ko_stats[t]['Final'] += 1
            
        champion = play_round(final_teams)[0]
        ko_stats[champion]['Win'] += 1
        
    print(f"\n🏆 TOURNAMENT PROGNOSE SUMMARY (using {model_name}) 🏆")
    print("=" * 60)
    print(f"{'Country':<20} | {'R16 %':<8} | {'QF %':<8} | {'SF %':<8} | {'Final %':<8} | {'Winner %':<8}")
    print("-" * 60)
    
    results = []
    for team, stats in ko_stats.items():
        results.append({
            'Team': team,
            'R16': (stats['R16'] / num_simulations) * 100,
            'QF': (stats['QF'] / num_simulations) * 100,
            'SF': (stats['SF'] / num_simulations) * 100,
            'Final': (stats['Final'] / num_simulations) * 100,
            'Win': (stats['Win'] / num_simulations) * 100
        })
        
    results_df = pd.DataFrame(results).sort_values(by='Win', ascending=False).head(15)
    for _, row in results_df.iterrows():
        print(f"{row['Team']:<20} | {row['R16']:>7.1f}% | {row['QF']:>7.1f}% | {row['SF']:>7.1f}% | {row['Final']:>7.1f}% | {row['Win']:>7.1f}%")
