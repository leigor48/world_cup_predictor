import pandas as pd
import os
import glob

def calculate_tournament_experience():
    tournaments_dir = os.path.join('data', 'raw', 'tournaments')
    squads_dir = os.path.join('data', 'processed', 'squads')
    
    tourn_files = glob.glob(os.path.join(tournaments_dir, '*.csv'))
    if not tourn_files:
        print("Kritischer Fehler: Keine Turnier-Daten in 'data/raw/tournaments/' gefunden.")
        return
        
    print("Lade und aggregiere historische Turnierdaten (WM & Kontinentalturniere)...")
    tourn_list = []
    
    for file in tourn_files:
        try:
            df = pd.read_csv(file)
            
            # Robuste Spaltensuche (Namen)
            col_name = 'player' if 'player' in df.columns else 'name' if 'name' in df.columns else None
            
            # Robuste Spaltensuche (Minuten)
            col_mins = None
            for col in df.columns:
                if 'minute' in col.lower() and 'play' in col.lower():
                    col_mins = col
                    break
            if not col_mins:
                for col in df.columns:
                    if 'minute' in col.lower() or 'mins' in col.lower():
                        col_mins = col
                        break
                        
            if not col_name or not col_mins:
                print(f"-> Überspringe {os.path.basename(file)}: Benötigte Spalten fehlen.")
                continue
                
            clean_df = df[[col_name, col_mins]].copy()
            clean_df.columns = ['player', 'minutes']
            
            clean_df['minutes'] = pd.to_numeric(clean_df['minutes'], errors='coerce').fillna(0)
            
            tourn_list.append(clean_df)
            
        except Exception as e:
            print(f"-> Fehler beim Verarbeiten von {file}: {e}")

    if not tourn_list:
        print("\nFehler: Turnier-Universum konnte nicht gebaut werden.")
        return

    tourn_df = pd.concat(tourn_list, ignore_index=True)
    tourn_df['match_name'] = tourn_df['player'].astype(str).str.lower().str.strip()
    
    # Minuten pro Spieler über alle vergangenen Turniere summieren
    tourn_agg = tourn_df.groupby('match_name')['minutes'].sum().reset_index()
    tourn_agg.rename(columns={'minutes': 'Total_Tournament_Minutes'}, inplace=True)
    
    print(f"Turnierdaten aggregiert: {len(tourn_agg)} Spieler mit int. Turniererfahrung.\n")
    print("Berechne Kader-Erfahrung...\n" + "-"*40)
    
    results = []
    squad_files = glob.glob(os.path.join(squads_dir, '*_cleaned.csv'))
    
    for file in squad_files:
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize()
        squad_df = pd.read_csv(file)
        
        squad_df['match_name'] = squad_df['Name'].astype(str).str.lower().str.strip()
        
        # Join
        merged = pd.merge(squad_df, tourn_agg, on='match_name', how='left')
        merged['Total_Tournament_Minutes'] = merged['Total_Tournament_Minutes'].fillna(0)
        
        squad_tourn_minutes = merged['Total_Tournament_Minutes'].sum()
        players_with_exp = (merged['Total_Tournament_Minutes'] > 0).sum()
        total_players = len(merged)
        
        print(f"{country}:")
        print(f"  -> {players_with_exp} von {total_players} Spielern waren bereits bei einer WM/EM/Copa auf dem Platz.")
        print(f"  -> Feature 'Historische Turnier-Minuten': {int(squad_tourn_minutes)}\n")
        
        results.append({
            'Country': country,
            'Total_Tournament_Minutes': int(squad_tourn_minutes)
        })

    features_dir = os.path.join('data', 'processed', 'features')
    os.makedirs(features_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(features_dir, 'tournament_experience.csv'), index=False)
    print("-> Feature 'Historische Turnier-Erfahrung' erfolgreich gespeichert.")

if __name__ == "__main__":
    calculate_tournament_experience()