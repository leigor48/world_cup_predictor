import ScraperFC
import pandas as pd
import os
import time

def scrape_historical_tournaments():
    sofa = ScraperFC.Sofascore()
    
    # Dictionary mit Turniername und den entsprechenden Jahren.
    # Wir nehmen die jeweils letzten beiden Ausgaben.
    tournaments = {
        'FIFA World Cup': ['2022', '2018'],
        'UEFA European Championship': ['2024', '2020'],
        'CONCACAF Gold Cup': ['2023', '2021'] # Nord-/Mittelamerika
    }
    
    # Ordnerstruktur sicherstellen
    raw_dir = os.path.join('data', 'raw', 'tournaments')
    os.makedirs(raw_dir, exist_ok=True)
    
    for tournament, years in tournaments.items():
        for year in years:
            print(f"Starte Scraping für {tournament} ({year})...")
            try:
                df = sofa.scrape_player_league_stats(year=year, league=tournament)
                
                filename = f"{tournament.replace(' ', '_').lower()}_{year}.csv"
                filepath = os.path.join(raw_dir, filename)
                
                df.to_csv(filepath, index=False)
                print(f"-> Erfolgreich gespeichert: {filepath} ({len(df)} Spieler)\n")
                
                # 5 Sekunden Pause für den Server
                time.sleep(5)
                
            except Exception as e:
                # Falls z.B. 2020 nicht als Jahr akzeptiert wird, sehen wir hier die validen Jahre
                print(f"-> Fehler bei {tournament} {year}: {e}\n")

if __name__ == "__main__":
    scrape_historical_tournaments()