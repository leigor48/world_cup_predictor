import pandas as pd
from curl_cffi import requests
from bs4 import BeautifulSoup
import os
import time

def scrape_fifa_ranking():
    print("Ziehe exakte Weltrangliste von Transfermarkt (via curl_cffi)...\n" + "-"*40)
    ranking_data = []
    
    # Top 75 Nationen holen
    for page in range(1, 4):
        url = f'https://www.transfermarkt.us/statistik/weltrangliste?page={page}'
        print(f"Lese Seite {page}...")
        
        try:
            response = requests.get(url, impersonate="chrome110", timeout=15)
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', class_='items')
            if not table: continue
                
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 5:
                    try:
                        rank_str = cols[0].text.strip()
                        team_td = row.find('td', class_='hauptlink')
                        if not team_td: continue
                        team_name = team_td.text.strip()
                        pts_str = cols[-1].text.strip().replace('.', '').replace(',', '')
                        
                        if rank_str.isdigit() and pts_str.isdigit():
                            ranking_data.append({
                                'Country': team_name,
                                'FIFA_Rank': int(rank_str),
                                'FIFA_Points': float(pts_str)
                            })
                    except Exception:
                        continue
            time.sleep(1.5)
        except Exception as e:
            print(f"Fehler auf Seite {page}: {e}")

    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        
        name_mapping = {
            'United States': 'USA', 
            'Korea, South': 'South Korea',
            'South Korea': 'Korea Republic',
            'Republic of Ireland': 'Ireland',
            'Cote d\'Ivoire': 'Ivory Coast'
        }
        ranking_df['Country'] = ranking_df['Country'].replace(name_mapping)
        
        features_dir = os.path.join('data', 'processed', 'features')
        os.makedirs(features_dir, exist_ok=True)
        ranking_df.to_csv(os.path.join(features_dir, 'fifa_ranking.csv'), index=False)
        print("\n-> FIFA Ranking erfolgreich als Feature gespeichert!\n")
    else:
        print("Fehler: Keine Daten extrahiert.")

if __name__ == "__main__":
    scrape_fifa_ranking()