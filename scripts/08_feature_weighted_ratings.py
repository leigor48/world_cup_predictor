import pandas as pd
import os
import glob

def calculate_weighted_ratings():
    leagues_dir = os.path.join('data', 'raw', 'leagues') 
    squads_dir = os.path.join('data', 'processed', 'squads')
    
    league_weights = {
        'england_premier_league': 1.00,
        'spain_la_liga': 0.95,
        'italy_serie_a': 0.95,
        'germany_bundesliga': 0.95,
        'france_ligue_1': 0.90,
        'netherlands_eredivisie': 0.80,
        'portugal_primeira_liga': 0.80,
        'brazil_serie_a': 0.75,
        'argentina_liga_profesional': 0.70,
        'turkiye_super_lig': 0.70,
        'england_efl_championship': 0.65,
        'germany_2.bundesliga': 0.60,
        'italy_serie_b': 0.60,
        'spain_la_liga_2': 0.60,
        'france_ligue_2': 0.55,
        'usa_mls': 0.55,
        'saudi_arabia_pro_league': 0.50
    }
    
    print("Lade und baue das Spieler-Universum...")
    universe_list = []
    
    for file in glob.glob(os.path.join(leagues_dir, '*.csv')): 
        if 'tm_' in file: continue 
        
        filename = os.path.basename(file).replace('.csv', '')
        league_key = filename.replace('_25_26', '').replace('_2026', '').replace('_2025', '')
        
        if league_key in league_weights:
            try:
                df = pd.read_csv(file)
                
                # Angepasste Suche nach der korrekten Spalte!
                col_name = 'player' if 'player' in df.columns else 'name' if 'name' in df.columns else None
                col_rating = 'rating' if 'rating' in df.columns else 'averageRating' if 'averageRating' in df.columns else None
                
                if not col_name or not col_rating:
                    print(f"-> Überspringe {league_key}: Benötigte Spalten fehlen.")
                    continue
                
                # DataFrame bauen
                clean_df = df[[col_name, col_rating]].copy()
                clean_df.columns = ['player', 'rating'] # Einheitlich benennen
                
                clean_df['League_Key'] = league_key
                clean_df['League_Weight'] = league_weights[league_key]
                
                universe_list.append(clean_df)
                
            except Exception as e:
                print(f"-> Fehler beim Verarbeiten von {file}: {e}")

    if not universe_list:
        print("\nKritischer Fehler: Universum konnte nicht gebaut werden.")
        return

    universe_df = pd.concat(universe_list, ignore_index=True)
    
    # Namen normalisieren
    universe_df['match_name'] = universe_df['player'].astype(str).str.lower().str.strip()
    universe_df = universe_df.dropna(subset=['rating'])
    
    # Die tatsächliche Berechnung: Rating * Faktor
    universe_df['Weighted_Rating'] = universe_df['rating'] * universe_df['League_Weight']
    
    print(f"Spieler-Universum erfolgreich geladen: {len(universe_df)} Datensätze.\n")
    print("Berechne Nationale Ratings...\n" + "-"*40)
    
    results = []
    
    squad_files = glob.glob(os.path.join(squads_dir, '*_cleaned.csv'))
    for file in squad_files:
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize()
        squad_df = pd.read_csv(file)
        
        squad_df['match_name'] = squad_df['Name'].astype(str).str.lower().str.strip()
        
        # Den Nationalkader mit den Liga-Ratings matchen
        merged = pd.merge(squad_df, universe_df, on='match_name', how='left')
        
        # Baseline für Spieler, die in keiner getrackten Liga spielen
        baseline_score = 6.5 * 0.3
        merged['Final_Weighted_Rating'] = merged['Weighted_Rating'].fillna(baseline_score)
        
        # Den Durchschnitt der besten 15 Spieler des Landes berechnen
        top_15_ratings = merged.nlargest(15, 'Final_Weighted_Rating')['Final_Weighted_Rating']
        squad_rating = top_15_ratings.mean()
        
        found_players = merged['Weighted_Rating'].notna().sum()
        total_players = len(merged)
        
        print(f"{country}:")
        print(f"  -> {found_players} von {total_players} Spielern im Daten-Universum gefunden.")
        print(f"  -> Feature 'Form-Rating (Top 15)': {squad_rating:.3f}\n")
        
        results.append({
            'Country': country,
            'Current_Form_Rating': squad_rating
        })

    features_dir = os.path.join('data', 'processed', 'features')
    os.makedirs(features_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(features_dir, 'current_form_ratings.csv'), index=False)
    print("-> Ratings erfolgreich gespeichert.")

if __name__ == "__main__":
    calculate_weighted_ratings()