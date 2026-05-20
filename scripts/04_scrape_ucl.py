import ScraperFC
import pandas as pd
import os
import time

def scrape_ucl_data():
    sofa = ScraperFC.Sofascore()
    
    # Wir ziehen die UCL der aktuellen sowie der letzten zwei Saisons
    tournament = 'UEFA Champions League'
    seasons = ['25/26', '24/25', '23/24']
    
    raw_dir = os.path.join('data', 'raw', 'ucl')
    os.makedirs(raw_dir, exist_ok=True)
    
    for season in seasons:
        print(f"Starte Scraping für {tournament} (Saison {season})...")
        try:
            # Gleiche Logik wie bei den Ligen
            df = sofa.scrape_player_league_stats(year=season, league=tournament)
            
            # Dateiname z.B. ucl_25_26.csv
            filename = f"ucl_{season.replace('/', '_')}.csv"
            filepath = os.path.join(raw_dir, filename)
            
            df.to_csv(filepath, index=False)
            print(f"-> Erfolgreich gespeichert: {filepath} ({len(df)} Spieler)\n")
            
            time.sleep(5)
            
        except Exception as e:
            print(f"-> Fehler bei {tournament} {season}: {e}\n")

if __name__ == "__main__":
    scrape_ucl_data()