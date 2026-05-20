import ScraperFC
import pandas as pd

def test_scraperfc():
    print("Initialisiere Sofascore Scraper...")
    sofa = ScraperFC.Sofascore()
    
    try:
        # Wir testen den Scraper, indem wir versuchen, die EPL-Statistiken eines Jahres abzurufen.
        # Wir nehmen ein abgelaufenes Jahr, da diese Daten meist stabil abrufbar sind.
        print("Versuche Daten für die Premier League (Saison 22/23) abzurufen...")
        
        # ScraperFC nutzt für Sofascore oft die Funktion scrape_league_stats oder scrape_player_league_stats
        # Achtung: Das Scrapen kann ein bis zwei Minuten dauern, da eine Browser-Instanz geöffnet wird.
        epl_stats = sofa.scrape_player_league_stats(year='22/23', league='England Premier League')        
        print("\nErfolgreich! Hier sind die ersten 5 Zeilen der Daten:")
        print(epl_stats.head())
        
        # Schliesst den Scraper sauber
        sofa.close()
        
    except Exception as e:
        print(f"\nFehler beim Scrapen aufgetreten: {e}")
        try:
            sofa.close()
        except:
            pass

if __name__ == "__main__":
    test_scraperfc()