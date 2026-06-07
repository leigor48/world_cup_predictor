import os
import glob
import time
import pandas as pd
from curl_cffi import requests as curl_requests
from bs4 import BeautifulSoup

def clean_market_values():
    """Cleans squad market value text and creates numerical columns and top-11 startelf features."""
    raw_dir = os.path.join('data', 'raw', 'squads')
    processed_dir = os.path.join('data', 'processed', 'squads')
    os.makedirs(processed_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(raw_dir, '*.csv'))
    if not files:
        print("Warning: No CSV files found in data/raw/squads.")
        return
        
    print("Processing squad market values...")
    for file in files:
        country = os.path.basename(file).replace('.csv', '').capitalize()
        df = pd.read_csv(file)
        
        def parse_value(val):
            if pd.isna(val) or val == '-':
                return 0.0
            val = str(val).replace('€', '').strip()
            if 'm' in val:
                return float(val.replace('m', ''))
            elif 'k' in val:
                return float(val.replace('k', '')) / 1000
            return 0.0
        
        df['Market_Value_mEUR'] = df['Market_Value'].apply(parse_value)
        
        save_path = os.path.join(processed_dir, f"{country.lower()}_cleaned.csv")
        df.to_csv(save_path, index=False)


def calculate_club_chemistry():
    """Calculates team chemistry index based on players playing for the same club."""
    processed_dir = os.path.join('data', 'processed', 'squads')
    files = glob.glob(os.path.join(processed_dir, '*_cleaned.csv'))
    
    if not files:
        print("Warning: No cleaned squad files found in data/processed/squads.")
        return
        
    print("Calculating club chemistry scores...")
    chemistry_results = []
    
    for file in files:
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize()
        country = country.replace('_', ' ')
        df = pd.read_csv(file)
        
        df_clubs = df[~df['Club'].isin(['Unknown', 'Without Club', 'Retired'])].copy()
        club_counts = df_clubs['Club'].value_counts()
        shared_clubs = club_counts[club_counts >= 2]
        
        total_chemistry_score = 0
        for club, n_players in shared_clubs.items():
            connections = int(n_players * (n_players - 1) / 2)
            total_chemistry_score += connections
            
        chemistry_results.append({
            'Country': country,
            'Chemistry_Score': total_chemistry_score
        })
        
    features_dir = os.path.join('data', 'processed', 'features')
    os.makedirs(features_dir, exist_ok=True)
    
    results_df = pd.DataFrame(chemistry_results)
    results_df.to_csv(os.path.join(features_dir, 'club_chemistry.csv'), index=False)
    print("-> Club chemistry calculations complete.")


def calculate_weighted_ratings():
    """Builds the player form rating weighted by league quality, and extracts European Top 5 League concentration."""
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
    
    top5_leagues = ['england_premier_league', 'spain_la_liga', 'italy_serie_a', 'germany_bundesliga', 'france_ligue_1']
    
    universe_list = []
    
    for file in glob.glob(os.path.join(leagues_dir, '*.csv')): 
        if 'tm_' in file: continue 
        filename = os.path.basename(file).replace('.csv', '')
        league_key = filename.replace('_25_26', '').replace('_2026', '').replace('_2025', '')
        
        if league_key in league_weights:
            try:
                df = pd.read_csv(file)
                col_name = 'player' if 'player' in df.columns else 'name' if 'name' in df.columns else None
                col_rating = 'rating' if 'rating' in df.columns else 'averageRating' if 'averageRating' in df.columns else None
                
                if not col_name or not col_rating:
                    continue
                
                clean_df = df[[col_name, col_rating]].copy()
                clean_df.columns = ['player', 'rating']
                clean_df['League_Key'] = league_key
                clean_df['League_Weight'] = league_weights[league_key]
                universe_list.append(clean_df)
            except Exception:
                pass

    if not universe_list:
        print("Warning: Player rating universe could not be constructed.")
        return

    universe_df = pd.concat(universe_list, ignore_index=True)
    universe_df['match_name'] = universe_df['player'].astype(str).str.lower().str.strip()
    universe_df = universe_df.dropna(subset=['rating'])
    universe_df['Weighted_Rating'] = universe_df['rating'] * universe_df['League_Weight']
    
    results = []
    squad_files = glob.glob(os.path.join(squads_dir, '*_cleaned.csv'))
    
    for file in squad_files:
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize()
        country = country.replace('_', ' ')
        squad_df = pd.read_csv(file)
        squad_df['match_name'] = squad_df['Name'].astype(str).str.lower().str.strip()
        
        # 1. Standard Form Rating Calculation
        merged = pd.merge(squad_df, universe_df, on='match_name', how='left')
        baseline_score = 6.5 * 0.3
        merged['Final_Weighted_Rating'] = merged['Weighted_Rating'].fillna(baseline_score)
        
        top_15_ratings = merged.nlargest(15, 'Final_Weighted_Rating')['Final_Weighted_Rating']
        squad_rating = top_15_ratings.mean()
        
        # 2. FINETUNING: Calculate Top-5 European League density
        # Clean Club name to avoid formatting mismatches and map to league keys
        # We can map known top-5 clubs or check in which leagues their name is matched.
        matched_leagues = merged.dropna(subset=['League_Key'])
        top5_count = matched_leagues[matched_leagues['League_Key'].isin(top5_leagues)]['match_name'].nunique()
        total_players = squad_df['match_name'].nunique()
        top5_density = (top5_count / total_players) if total_players > 0 else 0.0
        
        results.append({
            'Country': country,
            'Current_Form_Rating': squad_rating,
            'Top5_League_Density': top5_density
        })

    features_dir = os.path.join('data', 'processed', 'features')
    os.makedirs(features_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(features_dir, 'current_form_ratings.csv'), index=False)
    print("-> Weighted form ratings and Top-5 League Density features calculated.")


def calculate_ucl_experience():
    """Aggregates Champions League player minutes and maps to squad rosters."""
    ucl_dir = os.path.join('data', 'raw', 'ucl')
    squads_dir = os.path.join('data', 'processed', 'squads')
    
    ucl_files = glob.glob(os.path.join(ucl_dir, '*.csv'))
    if not ucl_files:
        print("Warning: No UCL files found in data/raw/ucl/.")
        return
        
    ucl_list = []
    for file in ucl_files:
        try:
            df = pd.read_csv(file)
            col_name = 'player' if 'player' in df.columns else 'name' if 'name' in df.columns else None
            
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
                continue
                
            clean_df = df[[col_name, col_mins]].copy()
            clean_df.columns = ['player', 'minutes']
            clean_df['minutes'] = pd.to_numeric(clean_df['minutes'], errors='coerce').fillna(0)
            ucl_list.append(clean_df)
        except Exception:
            pass

    if not ucl_list:
        print("Warning: No valid UCL data found.")
        return

    ucl_df = pd.concat(ucl_list, ignore_index=True)
    ucl_df['match_name'] = ucl_df['player'].astype(str).str.lower().str.strip()
    ucl_agg = ucl_df.groupby('match_name')['minutes'].sum().reset_index()
    ucl_agg.rename(columns={'minutes': 'Total_UCL_Minutes'}, inplace=True)
    
    results = []
    squad_files = glob.glob(os.path.join(squads_dir, '*_cleaned.csv'))
    
    for file in squad_files:
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize()
        country = country.replace('_', ' ')
        squad_df = pd.read_csv(file)
        squad_df['match_name'] = squad_df['Name'].astype(str).str.lower().str.strip()
        
        merged = pd.merge(squad_df, ucl_agg, on='match_name', how='left')
        merged['Total_UCL_Minutes'] = merged['Total_UCL_Minutes'].fillna(0)
        squad_ucl_minutes = merged['Total_UCL_Minutes'].sum()
        
        results.append({
            'Country': country,
            'Total_UCL_Minutes': int(squad_ucl_minutes)
        })

    features_dir = os.path.join('data', 'processed', 'features')
    os.makedirs(features_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(features_dir, 'ucl_experience.csv'), index=False)
    print("-> UCL experience feature calculated.")


def calculate_tournament_experience():
    """Aggregates World Cup/Continental tournament minutes and maps to rosters."""
    tournaments_dir = os.path.join('data', 'raw', 'tournaments')
    squads_dir = os.path.join('data', 'processed', 'squads')
    
    tourn_files = glob.glob(os.path.join(tournaments_dir, '*.csv'))
    if not tourn_files:
        print("Warning: No tournament files found in data/raw/tournaments/.")
        return
        
    tourn_list = []
    for file in tourn_files:
        try:
            df = pd.read_csv(file)
            col_name = 'player' if 'player' in df.columns else 'name' if 'name' in df.columns else None
            
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
                continue
                
            clean_df = df[[col_name, col_mins]].copy()
            clean_df.columns = ['player', 'minutes']
            clean_df['minutes'] = pd.to_numeric(clean_df['minutes'], errors='coerce').fillna(0)
            tourn_list.append(clean_df)
        except Exception:
            pass

    if not tourn_list:
        print("Warning: No valid historical tournament experience found.")
        return

    tourn_df = pd.concat(tourn_list, ignore_index=True)
    tourn_df['match_name'] = tourn_df['player'].astype(str).str.lower().str.strip()
    tourn_agg = tourn_df.groupby('match_name')['minutes'].sum().reset_index()
    tourn_agg.rename(columns={'minutes': 'Total_Tournament_Minutes'}, inplace=True)
    
    results = []
    squad_files = glob.glob(os.path.join(squads_dir, '*_cleaned.csv'))
    
    for file in squad_files:
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize()
        country = country.replace('_', ' ')
        squad_df = pd.read_csv(file)
        squad_df['match_name'] = squad_df['Name'].astype(str).str.lower().str.strip()
        
        merged = pd.merge(squad_df, tourn_agg, on='match_name', how='left')
        merged['Total_Tournament_Minutes'] = merged['Total_Tournament_Minutes'].fillna(0)
        squad_tourn_minutes = merged['Total_Tournament_Minutes'].sum()
        
        results.append({
            'Country': country,
            'Total_Tournament_Minutes': int(squad_tourn_minutes)
        })

    features_dir = os.path.join('data', 'processed', 'features')
    os.makedirs(features_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(features_dir, 'tournament_experience.csv'), index=False)
    print("-> Tournament experience feature calculated.")


def scrape_fifa_ranking():
    """Pulls current FIFA World Rankings from Transfermarkt."""
    print("Fetching FIFA Rankings from Transfermarkt...")
    ranking_data = []
    
    for page in range(1, 4):
        url = f'https://www.transfermarkt.us/statistik/weltrangliste?page={page}'
        try:
            response = curl_requests.get(url, impersonate="chrome110", timeout=15)
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', class_='items')
            if not table: 
                continue
                
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 5:
                    try:
                        rank_str = cols[0].text.strip()
                        team_td = row.find('td', class_='hauptlink')
                        if not team_td: 
                            continue
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
            print(f"Error scraping page {page}: {e}")

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
        print("-> FIFA rankings successfully extracted.")
    else:
        print("Error: No FIFA rankings could be extracted.")
