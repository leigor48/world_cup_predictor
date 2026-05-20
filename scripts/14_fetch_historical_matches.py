import pandas as pd
import os

def fetch_historical_matches():
    print("Lade historische Länderspiel-Daten herunter...\n" + "-"*40)
    
    # Direkter Raw-Link zum tagesaktuellen Kaggle-Dataset auf GitHub
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    
    try:
        df = pd.read_csv(url)
        
        # Datum in ein echtes Datetime-Format umwandeln
        df['date'] = pd.to_datetime(df['date'])
        
        # Wir nehmen nur Spiele ab 2018 (nach der WM in Russland), 
        # um den modernen Fussball abzubilden.
        df = df[df['date'].dt.year >= 2018].copy()
        
        # Wir berechnen das Ergebnis aus Sicht der Heimmannschaft (Team A)
        def get_outcome(row):
            if row['home_score'] > row['away_score']: return 2 # Sieg
            elif row['home_score'] == row['away_score']: return 1 # Draw
            else: return 0 # Niederlage
            
        df['target'] = df.apply(get_outcome, axis=1)
        
        # Spalten umbenennen, damit sie exakt zu unserem Matchup-Generator passen
        df.rename(columns={'home_team': 'Team_A', 'away_team': 'Team_B'}, inplace=True)
        
        # Relevante Spalten behalten
        final_df = df[['date', 'Team_A', 'Team_B', 'home_score', 'away_score', 'tournament', 'target']]
        
        # Speichern
        out_dir = os.path.join('data', 'raw', 'historical')
        os.makedirs(out_dir, exist_ok=True)
        
        out_path = os.path.join(out_dir, 'results.csv')
        final_df.to_csv(out_path, index=False)
        
        print(f"-> Erfolgreich {len(final_df)} historische Spiele (seit 2018) geladen und gespeichert!")
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\nVorschau der historischen Daten:")
        print(final_df.head())
        
    except Exception as e:
        print(f"Fehler beim Herunterladen der Daten: {e}")

if __name__ == "__main__":
    fetch_historical_matches()