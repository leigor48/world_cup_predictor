import os
import glob
import numpy as np
import pandas as pd
from thefuzz import fuzz, process

def clean_market_values():
    """Cleans squad market value text and prepares numerical columns."""
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
        df.to_csv(os.path.join(processed_dir, f"{country.lower()}_cleaned.csv"), index=False)


def calculate_squad_market_value():
    """Calculates logarithmically smoothed squad market value based on the Top 15 players."""
    processed_dir = os.path.join('data', 'processed', 'squads')
    files = glob.glob(os.path.join(processed_dir, '*_cleaned.csv'))
    
    print("Calculating logarithmic squad market values...")
    results = []
    
    for file in files:
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize().replace('_', ' ')
        df = pd.read_csv(file)
        
        # Top 15 reduziert Verzerrung durch riesige (oder zu kleine) Kadergrößen
        top_15_mv = df.nlargest(15, 'Market_Value_mEUR')['Market_Value_mEUR'].sum()
        
        # log1p glättet astronomische Summen (Unterschied 10->50 Mio ist spielerisch größer als 100->140 Mio)
        smoothed_mv = np.log1p(top_15_mv)
        
        results.append({
            'Country': country,
            'Log_Squad_Market_Value': smoothed_mv
        })
        
    features_dir = os.path.join('data', 'processed', 'features')
    os.makedirs(features_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(features_dir, 'squad_market_values.csv'), index=False)


def calculate_club_chemistry():
    """Calculates team chemistry index, weighted by the club's market value proxy."""
    processed_dir = os.path.join('data', 'processed', 'squads')
    files = glob.glob(os.path.join(processed_dir, '*_cleaned.csv'))
    
    print("Calculating quality-weighted club chemistry scores...")
    chemistry_results = []
    
    for file in files:
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize().replace('_', ' ')
        df = pd.read_csv(file)
        
        df_clubs = df[~df['Club'].isin(['Unknown', 'Without Club', 'Retired'])].copy()
        
        total_chemistry_score = 0.0
        for club, group in df_clubs.groupby('Club'):
            n_players = len(group)
            if n_players >= 2:
                connections = (n_players * (n_players - 1)) / 2
                
                # Gewichtung: Eine Achse von Real Madrid ist mehr wert als eine aus einer schwachen Liga
                club_quality_proxy = group['Market_Value_mEUR'].mean()
                weight = np.log1p(club_quality_proxy) if club_quality_proxy > 0 else 1.0
                
                total_chemistry_score += connections * weight
                
        chemistry_results.append({
            'Country': country,
            'Chemistry_Score': total_chemistry_score
        })
        
    features_dir = os.path.join('data', 'processed', 'features')
    pd.DataFrame(chemistry_results).to_csv(os.path.join(features_dir, 'club_chemistry.csv'), index=False)


def calculate_weighted_ratings():
    """Builds the player form rating weighted by global Opta-derived league indices."""
    leagues_dir = os.path.join('data', 'raw', 'leagues') 
    squads_dir = os.path.join('data', 'processed', 'squads')
    
    # Aktualisiert auf objektive, stufenweise Opta Power Rankings (skaliert auf PL = 1.0)
    league_weights = {
        'england_premier_league': 1.00,
        'spain_la_liga': 0.97,
        'italy_serie_a': 0.96,
        'germany_bundesliga': 0.95,
        'france_ligue_1': 0.93,
        'portugal_primeira_liga': 0.89,
        'netherlands_eredivisie': 0.87,
        'brazil_serie_a': 0.86,
        'england_efl_championship': 0.84,
        'argentina_liga_profesional': 0.83,
        'turkiye_super_lig': 0.82,
        'saudi_arabia_pro_league': 0.82,
        'usa_mls': 0.81,
        'spain_la_liga_2': 0.80,
        'germany_2.bundesliga': 0.79,
        'italy_serie_b': 0.78,
        'france_ligue_2': 0.77
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
                
                if col_name and col_rating:
                    clean_df = df[[col_name, col_rating]].copy()
                    clean_df.columns = ['player', 'rating']
                    clean_df['League_Key'] = league_key
                    clean_df['League_Weight'] = league_weights[league_key]
                    universe_list.append(clean_df)
            except Exception:
                pass

    if not universe_list:
        return

    universe_df = pd.concat(universe_list, ignore_index=True)
    universe_df['match_name'] = universe_df['player'].astype(str).str.lower().str.strip()
    universe_df = universe_df.dropna(subset=['rating'])
    universe_df['Weighted_Rating'] = universe_df['rating'] * universe_df['League_Weight']
    
    results = []
    
    for file in glob.glob(os.path.join(squads_dir, '*_cleaned.csv')):
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize().replace('_', ' ')
        squad_df = pd.read_csv(file)
        squad_df['match_name'] = squad_df['Name'].astype(str).str.lower().str.strip()
        
        merged = pd.merge(squad_df, universe_df, on='match_name', how='left')
        baseline_score = 6.5 * 0.5 # Leicht erhöhte Baseline für Nationalspieler ohne Ligadaten
        merged['Final_Weighted_Rating'] = merged['Weighted_Rating'].fillna(baseline_score)
        
        squad_rating = merged.nlargest(15, 'Final_Weighted_Rating')['Final_Weighted_Rating'].mean()
        
        matched_leagues = merged.dropna(subset=['League_Key'])
        top5_count = matched_leagues[matched_leagues['League_Key'].isin(top5_leagues)]['match_name'].nunique()
        total_players = len(squad_df)
        top5_density = (top5_count / total_players) if total_players > 0 else 0.0
        
        results.append({
            'Country': country,
            'Current_Form_Rating': squad_rating,
            'Top5_League_Density': top5_density
        })

    features_dir = os.path.join('data', 'processed', 'features')
    pd.DataFrame(results).to_csv(os.path.join(features_dir, 'current_form_ratings.csv'), index=False)


def get_fuzzy_minutes(squad_df, minutes_df, source_col='minutes', target_col='minutes', threshold=85):
    """Helper function to fuzzy match player names between datasets."""
    minutes_dict = {}
    minutes_names = minutes_df['match_name'].tolist()
    
    if not minutes_names:
        squad_df[target_col] = 0.0
        return squad_df
        
    for s_name in squad_df['match_name']:
        match, score = process.extractOne(s_name, minutes_names, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            minutes_dict[s_name] = minutes_df.loc[minutes_df['match_name'] == match, source_col].values[0]
        else:
            minutes_dict[s_name] = 0.0
            
    squad_df[target_col] = squad_df['match_name'].map(minutes_dict)
    return squad_df


def calculate_ucl_experience():
    """Aggregates Champions League player minutes with robust name matching."""
    ucl_dir = os.path.join('data', 'raw', 'ucl')
    squads_dir = os.path.join('data', 'processed', 'squads')
    
    print("Calculating UCL experience (Fuzzy Matching)...")
    ucl_files = glob.glob(os.path.join(ucl_dir, '*.csv'))
    
    ucl_list = []
    for file in ucl_files:
        try:
            df = pd.read_csv(file)
            col_name = 'player' if 'player' in df.columns else 'name' if 'name' in df.columns else None
            col_mins = next((c for c in df.columns if 'minute' in c.lower() or 'mins' in c.lower()), None)
            
            if col_name and col_mins:
                clean_df = df[[col_name, col_mins]].copy()
                clean_df.columns = ['player', 'minutes']
                clean_df['minutes'] = pd.to_numeric(clean_df['minutes'], errors='coerce').fillna(0)
                ucl_list.append(clean_df)
        except Exception:
            pass

    if not ucl_list:
        return

    ucl_df = pd.concat(ucl_list, ignore_index=True)
    ucl_df['match_name'] = ucl_df['player'].astype(str).str.lower().str.strip()
    ucl_agg = ucl_df.groupby('match_name')['minutes'].sum().reset_index()
    
    results = []
    for file in glob.glob(os.path.join(squads_dir, '*_cleaned.csv')):
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize().replace('_', ' ')
        squad_df = pd.read_csv(file)
        squad_df['match_name'] = squad_df['Name'].astype(str).str.lower().str.strip()
        
        merged = get_fuzzy_minutes(squad_df, ucl_agg, source_col='minutes', target_col='ucl_mins')
        
        # Durchschnitt der 15 erfahrensten Spieler verhindert Kadergrößen/Überalterungs-Bias
        squad_ucl_minutes = merged.nlargest(15, 'ucl_mins')['ucl_mins'].mean()
        
        results.append({
            'Country': country,
            'Total_UCL_Minutes': squad_ucl_minutes
        })

    features_dir = os.path.join('data', 'processed', 'features')
    pd.DataFrame(results).to_csv(os.path.join(features_dir, 'ucl_experience.csv'), index=False)


def calculate_tournament_experience():
    """Aggregates World Cup/Continental tournament minutes with robust name matching."""
    tournaments_dir = os.path.join('data', 'raw', 'tournaments')
    squads_dir = os.path.join('data', 'processed', 'squads')
    
    print("Calculating historical tournament experience (Fuzzy Matching)...")
    tourn_files = glob.glob(os.path.join(tournaments_dir, '*.csv'))
    
    tourn_list = []
    for file in tourn_files:
        try:
            df = pd.read_csv(file)
            col_name = 'player' if 'player' in df.columns else 'name' if 'name' in df.columns else None
            col_mins = next((c for c in df.columns if 'minute' in c.lower() or 'mins' in c.lower()), None)
            
            if col_name and col_mins:
                clean_df = df[[col_name, col_mins]].copy()
                clean_df.columns = ['player', 'minutes']
                clean_df['minutes'] = pd.to_numeric(clean_df['minutes'], errors='coerce').fillna(0)
                tourn_list.append(clean_df)
        except Exception:
            pass

    if not tourn_list:
        return

    tourn_df = pd.concat(tourn_list, ignore_index=True)
    tourn_df['match_name'] = tourn_df['player'].astype(str).str.lower().str.strip()
    tourn_agg = tourn_df.groupby('match_name')['minutes'].sum().reset_index()
    
    results = []
    for file in glob.glob(os.path.join(squads_dir, '*_cleaned.csv')):
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize().replace('_', ' ')
        squad_df = pd.read_csv(file)
        squad_df['match_name'] = squad_df['Name'].astype(str).str.lower().str.strip()
        
        merged = get_fuzzy_minutes(squad_df, tourn_agg, source_col='minutes', target_col='tourn_mins')
        squad_tourn_minutes = merged.nlargest(15, 'tourn_mins')['tourn_mins'].mean()
        
        results.append({
            'Country': country,
            'Total_Tournament_Minutes': squad_tourn_minutes
        })

    features_dir = os.path.join('data', 'processed', 'features')
    pd.DataFrame(results).to_csv(os.path.join(features_dir, 'tournament_experience.csv'), index=False)