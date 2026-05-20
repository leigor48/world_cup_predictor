import pandas as pd
import os
import glob

def calculate_club_chemistry():
    processed_dir = os.path.join('data', 'processed', 'squads')
    files = glob.glob(os.path.join(processed_dir, '*_cleaned.csv'))
    
    if not files:
        print("Keine bereinigten Kader-Daten gefunden.")
        return
        
    print("Berechne Club-Chemistry-Index...\n" + "-"*40)
    
    chemistry_results = []
    
    for file in files:
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize()
        df = pd.read_csv(file)
        
        # Ignoriere Spieler ohne Verein
        df_clubs = df[~df['Club'].isin(['Unknown', 'Without Club', 'Retired'])].copy()
        
        # Zähle, wie viele Spieler in welchem Verein spielen
        club_counts = df_clubs['Club'].value_counts()
        
        # Filtere Vereine, bei denen mindestens 2 Nationalspieler spielen
        shared_clubs = club_counts[club_counts >= 2]
        
        total_chemistry_score = 0
        details = []
        
        for club, n_players in shared_clubs.items():
            # Kombinatorik: n * (n - 1) / 2
            connections = int(n_players * (n_players - 1) / 2)
            total_chemistry_score += connections
            
            # Für die Nachvollziehbarkeit speichern wir, welche Vereine Blöcke bilden
            spieler_im_club = df_clubs[df_clubs['Club'] == club]['Name'].tolist()
            details.append(f"{club} ({n_players} Spieler: {', '.join(spieler_im_club)}) -> {connections} Links")
            
        chemistry_results.append({
            'Country': country,
            'Chemistry_Score': total_chemistry_score
        })
        
        print(f"{country} - Chemistry Score: {total_chemistry_score}")
        for detail in details:
            print(f"  * {detail}")
        print()
        
    # Speichere die Resultate in einer neuen Übersichtstabelle
    features_dir = os.path.join('data', 'processed', 'features')
    os.makedirs(features_dir, exist_ok=True)
    
    results_df = pd.DataFrame(chemistry_results)
    results_df.to_csv(os.path.join(features_dir, 'club_chemistry.csv'), index=False)
    print(f"-> Chemistry-Scores erfolgreich unter 'data/processed/features/club_chemistry.csv' gespeichert.")

if __name__ == "__main__":
    calculate_club_chemistry()