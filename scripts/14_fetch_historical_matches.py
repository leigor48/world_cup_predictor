import pandas as pd
import os

def fetch_historical_matches():
    print("Lade historische Länderspiel-Daten herunter...\n" + "-"*40)
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    
    try:
        df = pd.read_csv(url)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year >= 2018].copy()
        
        def get_outcome(row):
            if row['home_score'] > row['away_score']: return 2
            elif row['home_score'] == row['away_score']: return 1
            else: return 0
            
        df['target'] = df.apply(get_outcome, axis=1)
        df.rename(columns={'home_team': 'Team_A', 'away_team': 'Team_B'}, inplace=True)
        
        # NEU: Wir behalten die Spalte 'neutral'
        final_df = df[['date', 'Team_A', 'Team_B', 'home_score', 'away_score', 'tournament', 'neutral', 'target']]
        
        out_dir = os.path.join('data', 'raw', 'historical')
        os.makedirs(out_dir, exist_ok=True)
        final_df.to_csv(os.path.join(out_dir, 'results.csv'), index=False)
        
        print(f"-> Erfolgreich {len(final_df)} historische Spiele inkl. Heimvorteil-Flag geladen!")
    except Exception as e:
        print(f"Fehler: {e}")

if __name__ == "__main__":
    fetch_historical_matches()