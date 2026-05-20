import ScraperFC
import pandas as pd
import os
import time

def scrape_all_leagues():
    sofa = ScraperFC.Sofascore()
    
    # Dictionary mit Liga und dem entsprechenden korrekten Saison-Format
    league_seasons = {
        # Top 5
        'England Premier League': '25/26', 
        'Spain La Liga': '25/26', 
        'Italy Serie A': '25/26', 
        'Germany Bundesliga': '25/26', 
        'France Ligue 1': '25/26',
        
        # Weitere europäische Top-Ligen
        'Netherlands Eredivisie': '25/26', 
        'Portugal Primeira Liga': '25/26', 
        'Turkiye Super Lig': '25/26',
        
        # Wichtige aussereuropäische Ligen (Kalenderjahr-Format!)
        'USA MLS': '2026', 
        'Saudi Arabia Pro League': '25/26', 
        'Argentina Liga Profesional': '2026',
        
        # Zweite Ligen
        'England EFL Championship': '25/26', 
        'Germany 2.Bundesliga': '25/26', 
        'Spain La Liga 2': '25/26', 
        'Italy Serie B': '25/26', 
        'France Ligue 2': '25/26'
    }
    
    os.makedirs(os.path.join('data', 'raw'), exist_ok=True)
    
    for league, season in league_seasons.items():
        print(f"Starte Scraping für {league} (Saison {season})...")
        try:
            df = sofa.scrape_player_league_stats(year=season, league=league)
            
            filename = f"{league.replace(' ', '_').lower()}_{season.replace('/', '_')}.csv"
            filepath = os.path.join('data', 'raw', filename)
            
            df.to_csv(filepath, index=False)
            print(f"-> Erfolgreich: {filepath} ({len(df)} Spieler)\n")
            
            time.sleep(5)
            
        except Exception as e:
            print(f"-> Fehler bei {league}: {e}\n")

if __name__ == "__main__":
    scrape_all_leagues()