import pandas as pd
import os
import itertools

def create_matchups():
    master_path = os.path.join('data', 'processed', 'features', 'MASTER_dataset.csv')
    
    if not os.path.exists(master_path):
        print("Fehler: MASTER_dataset.csv nicht gefunden.")
        return
        
    df = pd.read_csv(master_path)
    print("Erstelle Matchup-Matrix (Team A vs Team B)...\n" + "-"*40)
    
    matchups = []
    
    # Wir erstellen alle möglichen Spiel-Kombinationen unserer Test-Nationen
    teams = df['Country'].tolist()
    combinations = list(itertools.combinations(teams, 2))
    
    for team_a, team_b in combinations:
        data_a = df[df['Country'] == team_a].iloc[0]
        data_b = df[df['Country'] == team_b].iloc[0]
        
        # Die Delta-Features berechnen (Team A - Team B)
        # Hinweis: Bei Rankings (Platz 1 vs Platz 10) ist ein negatives Delta eigentlich "besser", 
        # aber das ML-Modell lernt diese negative Korrelation ganz automatisch.
        matchup_data = {
            'Team_A': team_a,
            'Team_B': team_b,
            'Delta_Total_Market_Value': data_a['Total_Market_Value_mEUR'] - data_b['Total_Market_Value_mEUR'],
            'Delta_Median_Top11_Value': data_a['Median_Top11_Market_Value_mEUR'] - data_b['Median_Top11_Market_Value_mEUR'],
            'Delta_Chemistry': data_a['Chemistry_Score'] - data_b['Chemistry_Score'],
            'Delta_Form_Rating': data_a['Current_Form_Rating'] - data_b['Current_Form_Rating'],
            'Delta_UCL_Minutes': data_a['Total_UCL_Minutes'] - data_b['Total_UCL_Minutes'],
            'Delta_Tournament_Minutes': data_a['Total_Tournament_Minutes'] - data_b['Total_Tournament_Minutes'],
            'Delta_Average_Age': data_a['Average_Age'] - data_b['Average_Age'],
            'Delta_TM_Value_Rank': data_a['TM_Value_Rank'] - data_b['TM_Value_Rank'],
            'Delta_FIFA_Rank': data_a['FIFA_Rank'] - data_b['FIFA_Rank'],
            'Delta_FIFA_Points': data_a['FIFA_Points'] - data_b['FIFA_Points']
        }
        
        matchups.append(matchup_data)
        
    matchup_df = pd.DataFrame(matchups)
    
    # Speichern der Matchup-Matrix im neuen Ordner für den Model-Input
    out_dir = os.path.join('data', 'processed', 'model_input')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'matchup_features.csv')
    matchup_df.to_csv(out_path, index=False)
    
    print(f"-> Erfolgreich {len(combinations)} Test-Matchups generiert und gespeichert!")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nVorschau der Matchup-Deltas (aus Sicht von Team A):")
    print(matchup_df[['Team_A', 'Team_B', 'Delta_Total_Market_Value', 'Delta_FIFA_Points', 'Delta_Chemistry']].head())

if __name__ == "__main__":
    create_matchups()