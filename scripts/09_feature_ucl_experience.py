import pandas as pd
import os
import glob

def calculate_ucl_experience():
    ucl_dir = os.path.join('data', 'raw', 'ucl')
    squads_dir = os.path.join('data', 'processed', 'squads')
    
    ucl_files = glob.glob(os.path.join(ucl_dir, '*.csv'))
    if not ucl_files:
        print("Kritischer Fehler: Keine UCL-Daten in 'data/raw/ucl/' gefunden.")
        return
        
    print("Lade und aggregiere Champions-League-Daten (letzte 3 Saisons)...")
    ucl_list = []
    
    for file in ucl_files:
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
            
            # Sicherstellen, dass Minuten Zahlen sind (z.B. falls ein "90'" als String kommt)
            clean_df['minutes'] = pd.to_numeric(clean_df['minutes'], errors='coerce').fillna(0)
            
            ucl_list.append(clean_df)
            
        except Exception as e:
            print(f"-> Fehler beim Verarbeiten von {file}: {e}")

    if not ucl_list:
        print("\nFehler: UCL-Universum konnte nicht gebaut werden.")
        return

    # Alle 3 Saisons untereinander hängen
    ucl_df = pd.concat(ucl_list, ignore_index=True)
    ucl_df['match_name'] = ucl_df['player'].astype(str).str.lower().str.strip()
    
    # Minuten pro Spieler über alle 3 Saisons summieren
    ucl_agg = ucl_df.groupby('match_name')['minutes'].sum().reset_index()
    ucl_agg.rename(columns={'minutes': 'Total_UCL_Minutes'}, inplace=True)
    
    print(f"UCL-Daten aggregiert: {len(ucl_agg)} Spieler mit UCL-Erfahrung weltweit.\n")
    print("Berechne Kader-Erfahrung...\n" + "-"*40)
    
    results = []
    squad_files = glob.glob(os.path.join(squads_dir, '*_cleaned.csv'))
    
    for file in squad_files:
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize()
        squad_df = pd.read_csv(file)
        
        squad_df['match_name'] = squad_df['Name'].astype(str).str.lower().str.strip()
        
        # Join: Spieler im Kader mit ihren UCL-Minuten verknüpfen
        merged = pd.merge(squad_df, ucl_agg, on='match_name', how='left')
        
        # Wer nicht in der UCL gespielt hat, bekommt 0 Minuten
        merged['Total_UCL_Minutes'] = merged['Total_UCL_Minutes'].fillna(0)
        
        # Gesamte UCL-Erfahrung des Kaders summieren
        squad_ucl_minutes = merged['Total_UCL_Minutes'].sum()
        
        # Zählen, wie viele Spieler überhaupt UCL-Erfahrung haben (fürs Terminal)
        players_with_ucl = (merged['Total_UCL_Minutes'] > 0).sum()
        total_players = len(merged)
        
        print(f"{country}:")
        print(f"  -> {players_with_ucl} von {total_players} Spielern haben UCL-Erfahrung.")
        print(f"  -> Feature 'Gesamte UCL-Minuten': {int(squad_ucl_minutes)}\n")
        
        results.append({
            'Country': country,
            'Total_UCL_Minutes': int(squad_ucl_minutes)
        })

    features_dir = os.path.join('data', 'processed', 'features')
    os.makedirs(features_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(features_dir, 'ucl_experience.csv'), index=False)
    print("-> Feature 'UCL-Erfahrung' erfolgreich gespeichert.")

if __name__ == "__main__":
    calculate_ucl_experience()