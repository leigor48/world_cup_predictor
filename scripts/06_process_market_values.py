import pandas as pd
import os
import glob

def clean_market_values():
    # Definiere die Pfade
    raw_dir = os.path.join('data', 'raw', 'squads')
    
    # Neuer Ordner für verarbeitete Daten (Feature Engineering)
    processed_dir = os.path.join('data', 'processed', 'squads')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Lade alle CSV-Dateien aus unserem Squads-Ordner
    files = glob.glob(os.path.join(raw_dir, '*.csv'))
    
    if not files:
        print("Keine CSV-Dateien im raw/squads Ordner gefunden.")
        return
        
    print("Starte Feature-Engineering für Marktwerte...\n" + "-"*40)
    
    for file in files:
        # Dateinamen extrahieren (z.B. switzerland)
        country = os.path.basename(file).replace('.csv', '').capitalize()
        df = pd.read_csv(file)
        
        # 1. Funktion zur Umwandlung des Textes in Float (Millionen Euro)
        def parse_value(val):
            if pd.isna(val) or val == '-':
                return 0.0
            
            val = str(val).replace('€', '').strip()
            
            if 'm' in val:
                return float(val.replace('m', ''))
            elif 'k' in val:
                # Tausender in Millionen umrechnen (z.B. 800k = 0.8m)
                return float(val.replace('k', '')) / 1000
            
            return 0.0
        
        # 2. Neue numerische Spalte erstellen
        df['Market_Value_mEUR'] = df['Market_Value'].apply(parse_value)
        
        # 3. Speichern der bereinigten Daten
        save_path = os.path.join(processed_dir, f"{country.lower()}_cleaned.csv")
        df.to_csv(save_path, index=False)
        
        # 4. Feature Berechnung: Median der Top-11 Spieler (voraussichtliche Startelf)
        top_11 = df.nlargest(11, 'Market_Value_mEUR')
        median_top11 = top_11['Market_Value_mEUR'].median()
        total_value = df['Market_Value_mEUR'].sum()
        
        print(f"{country}:")
        print(f"  -> Gesamter Kaderwert: {total_value:.2f} Mio. €")
        print(f"  -> Feature 'Median Startelf': {median_top11:.2f} Mio. €\n")

if __name__ == "__main__":
    clean_market_values()