import pandas as pd
import numpy as np
import joblib
import os
import itertools

def simulate_groups():
    # 1. Modell und Master-Daten laden
    model_path = os.path.join('models', 'xgboost_wm_modelV3.joblib')
    master_path = os.path.join('data', 'processed', 'features', 'MASTER_dataset.csv')
    
    if not os.path.exists(model_path) or not os.path.exists(master_path):
        print("Fehler: Modell oder MASTER_dataset fehlt.")
        return
        
    model = joblib.load(model_path)
    master_df = pd.read_csv(master_path)
    
    # 2. Unsere offiziellen 12 WM-Gruppen
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
    
    # Exakte Feature-Reihenfolge wie beim Training!
    features = [
        'Delta_Total_Market_Value', 'Delta_Median_Top11_Value', 'Delta_Chemistry',
        'Delta_Form_Rating', 'Delta_UCL_Minutes', 'Delta_Tournament_Minutes',
        'Delta_Average_Age', 'Delta_TM_Value_Rank', 'Delta_FIFA_Rank', 'Delta_FIFA_Points',
        'Is_Neutral'
    ]
    
    print("🏆 STARTE WM-GRUPPENPHASEN-SIMULATION 🏆\n" + "="*45)
    
    # 3. Gruppen durchspielen
    for group_name, teams in groups.items():
        # Tabelle initialisieren (Punkte auf 0 setzen)
        group_standings = {team: 0 for team in teams}
        print(f"\n--- GRUPPE {group_name} ---")
        
        # Alle möglichen Matchups in der Gruppe (Jeder gegen Jeden)
        matchups = list(itertools.combinations(teams, 2))
        
        for team_a, team_b in matchups:
            # Daten für beide Teams holen (case-insensitive für absolute Sicherheit)
            data_a = master_df[master_df['Country'].str.lower() == team_a.lower()]
            data_b = master_df[master_df['Country'].str.lower() == team_b.lower()]
            
            if data_a.empty or data_b.empty:
                print(f"⚠️ Warnung: {team_a} oder {team_b} nicht im Datensatz gefunden!")
                continue
                
            data_a = data_a.iloc[0]
            data_b = data_b.iloc[0]
            
            # Deltas berechnen
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
                'Is_Neutral': 1 # Es ist eine WM, also immer neutraler Boden!
            }
            
            # DataFrame für die Vorhersage bauen
            X_pred = pd.DataFrame([delta_dict])[features]
            
            # Vorhersage treffen
            probs = model.predict_proba(X_pred)[0]
            outcome = np.random.choice([0, 1, 2], p=probs)
            
            if outcome == 2:
                print(f"{team_a} besiegt {team_b} (Sieg-Wahrsch: {probs[2]*100:.1f}%)")
                group_standings[team_a] += 3
            elif outcome == 0:
                print(f"{team_b} besiegt {team_a} (Sieg-Wahrsch: {probs[0]*100:.1f}%)")
                group_standings[team_b] += 3
            else:
                print(f"{team_a} und {team_b} trennen sich Unentschieden (Wahrsch: {probs[1]*100:.1f}%)")
                group_standings[team_a] += 1
                group_standings[team_b] += 1
                
        # 4. Tabelle sortieren und ausgeben
        # Bei Punktgleichheit entscheidet hier vorerst die pure FIFA-Punktzahl als Tie-Breaker
        sorted_standings = sorted(
            group_standings.items(), 
            key=lambda x: (x[1], master_df[master_df['Country'].str.lower() == x[0].lower()].iloc[0]['FIFA_Points']), 
            reverse=True
        )
        
        print("\nEndstand Tabelle:")
        for rank, (team, points) in enumerate(sorted_standings, 1):
            print(f"{rank}. {team:<20} {points} Pkt.")

if __name__ == "__main__":
    simulate_groups()