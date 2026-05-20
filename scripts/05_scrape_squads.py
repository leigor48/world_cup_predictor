import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import re  # WICHTIG: Für Regular Expressions

def scrape_tm_national_squad(country, url, save_dir):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    
    print(f"Ziehe Kader & Marktwerte für {country}...")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"-> Fehler: Transfermarkt hat mit Status Code {response.status_code} geantwortet.")
        return
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    table = soup.find('table', {'class': 'items'})
    if not table:
        print(f"-> Fehler: Konnte die Kadertabelle für {country} nicht finden.")
        return
    
    players = []
    rows = table.find('tbody').find_all('tr', class_=['odd', 'even'])
    
    for row in rows:
        try:
            # 1. Name
            name_tag = row.find('td', class_='hauptlink')
            if not name_tag:
                continue
            name = name_tag.text.strip()
            
            # 2. Position
            inline_table = row.find('table', class_='inline-table')
            if inline_table:
                position = inline_table.find_all('tr')[-1].text.strip()
            else:
                position = "Unknown"
            
            # 3. VEREIN (ROBUSTE LOGIK)
            club = "Unknown"
            # Wir suchen gezielt nach einem Link, der '/verein/' enthält
            club_link = row.find('a', href=re.compile(r'/verein/\d+'))
            if club_link:
                # Meistens steht der Vereinsname im 'title' Attribut des Links
                if 'title' in club_link.attrs and club_link['title']:
                    club = club_link['title']
                else:
                    # Alternativ im 'alt' Attribut des Logos
                    img = club_link.find('img')
                    if img and 'alt' in img.attrs:
                        club = img['alt']
            
            # 4. Marktwert
            value_td = row.find_all('td', class_='rechts')[-1]
            market_value_str = value_td.text.strip()
            
            players.append({
                'Name': name,
                'Position': position,
                'Club': club,
                'Market_Value': market_value_str
            })
            
        except Exception as e:
            continue
            
    df = pd.DataFrame(players)
    filename = f"{country.lower()}.csv"
    filepath = os.path.join(save_dir, filename)
    df.to_csv(filepath, index=False)
    
    print(f"-> Erfolgreich: {len(df)} Spieler für {country} gespeichert.")

def main():
    # Unsere Test-Nationen (später erweitern wir dieses Dictionary auf alle 48 WM-Teilnehmer)
    national_teams = {
        # --- Gruppe A ---
        'Mexico': 'https://www.transfermarkt.com/mexiko/kader/verein/6303',
        'South Africa': 'https://www.transfermarkt.com/sudafrika/kader/verein/3385',
        'South Korea': 'https://www.transfermarkt.com/sudkorea/kader/verein/3589',
        'Czech Republic': 'https://www.transfermarkt.com/tschechien/kader/verein/3445',

        # --- Gruppe B ---
        'Canada': 'https://www.transfermarkt.com/kanada/kader/verein/3510',
        'Bosnia and Herzegovina': 'https://www.transfermarkt.com/bosnien-herzegowina/kader/verein/3446',
        'Qatar': 'https://www.transfermarkt.com/katar/kader/verein/14162',
        'Switzerland': 'https://www.transfermarkt.com/schweiz/kader/verein/3384',

        # --- Gruppe C ---
        'Brazil': 'https://www.transfermarkt.com/brasilien/kader/verein/3439',
        'Morocco': 'https://www.transfermarkt.com/marokko/kader/verein/3575',
        'Haiti': 'https://www.transfermarkt.com/haiti/kader/verein/14161',
        'Scotland': 'https://www.transfermarkt.com/schottland/kader/verein/3382',

        # --- Gruppe D ---
        'USA': 'https://www.transfermarkt.com/vereinigte-staaten/kader/verein/3505',
        'Paraguay': 'https://www.transfermarkt.com/paraguay/kader/verein/3581',
        'Australia': 'https://www.transfermarkt.com/australien/kader/verein/3433',
        'Turkey': 'https://www.transfermarkt.com/turkei/kader/verein/3381',

        # --- Gruppe E ---
        'Germany': 'https://www.transfermarkt.com/deutschland/kader/verein/3262',
        'Curacao': 'https://www.transfermarkt.com/curacao/kader/verein/32364',
        'Ivory Coast': 'https://www.transfermarkt.com/elfenbeinkuste/kader/verein/3591',
        'Ecuador': 'https://www.transfermarkt.com/ecuador/kader/verein/5784',

        # --- Gruppe F ---
        'Netherlands': 'https://www.transfermarkt.com/niederlande/kader/verein/3379',
        'Japan': 'https://www.transfermarkt.com/japan/kader/verein/3435',
        'Sweden': 'https://www.transfermarkt.com/schweden/kader/verein/3557',
        'Tunisia': 'https://www.transfermarkt.com/tunesien/kader/verein/3670',

        # --- Gruppe G ---
        'Belgium': 'https://www.transfermarkt.com/belgien/kader/verein/3376',
        'Egypt': 'https://www.transfermarkt.com/agypten/kader/verein/3672',
        'Iran': 'https://www.transfermarkt.com/iran/kader/verein/3582',
        'New Zealand': 'https://www.transfermarkt.com/neuseeland/kader/verein/3588',

        # --- Gruppe H ---
        'Spain': 'https://www.transfermarkt.com/spanien/kader/verein/3375',
        'Cape Verde': 'https://www.transfermarkt.com/kap-verde/kader/verein/4311',
        'Saudi Arabia': 'https://www.transfermarkt.com/saudi-arabien/kader/verein/3807',
        'Uruguay': 'https://www.transfermarkt.com/uruguay/kader/verein/3449',

        # --- Gruppe I ---
        'France': 'https://www.transfermarkt.com/frankreich/kader/verein/3377',
        'Senegal': 'https://www.transfermarkt.com/senegal/kader/verein/3499',
        'Iraq': 'https://www.transfermarkt.com/irak/kader/verein/3560',
        'Norway': 'https://www.transfermarkt.com/norwegen/kader/verein/3436',

        # --- Gruppe J ---
        'Argentina': 'https://www.transfermarkt.com/argentinien/kader/verein/3437',
        'Algeria': 'https://www.transfermarkt.com/algerien/kader/verein/3614',
        'Austria': 'https://www.transfermarkt.com/osterreich/kader/verein/3383',
        'Jordan': 'https://www.transfermarkt.com/jordanien/kader/verein/15737',

        # --- Gruppe K ---
        'Portugal': 'https://www.transfermarkt.com/portugal/kader/verein/3300',
        'DR Congo': 'https://www.transfermarkt.com/dr-kongo/kader/verein/3854',
        'Uzbekistan': 'https://www.transfermarkt.com/usbekistan/kader/verein/3563',
        'Colombia': 'https://www.transfermarkt.com/kolumbien/kader/verein/3816',

        # --- Gruppe L ---
        'England': 'https://www.transfermarkt.com/england/kader/verein/3299',
        'Croatia': 'https://www.transfermarkt.com/kroatien/kader/verein/3556',
        'Ghana': 'https://www.transfermarkt.com/ghana/kader/verein/3441',
        'Panama': 'https://www.transfermarkt.com/panama/kader/verein/3577'
    }
    
    raw_dir = os.path.join('data', 'raw', 'squads')
    os.makedirs(raw_dir, exist_ok=True)
    
    for country, url in national_teams.items():
        scrape_tm_national_squad(country, url, raw_dir)
        # Respektvolle Pause, da wir keinen API-Key nutzen
        time.sleep(3)

if __name__ == "__main__":
    main()