import os
import numpy as np
import pandas as pd

def calculate_elo_ratings():
    """Chronologically calculates mathematically stable ELO ratings with string-cleaning and anti-farming caps."""
    results_path = os.path.join('data', 'raw', 'historical', 'results.csv')
    if not os.path.exists(results_path):
        print(f"Error: Historical results CSV not found at {results_path}. Run scraping step first.")
        return
        
    df = pd.read_csv(results_path)
    
    # 1. Datenbereinigung: Verhindert gesplittete Punkte durch Leerzeichen oder Groß-/Kleinschreibung
    df['Team_A'] = df['Team_A'].astype(str).str.strip().str.title()
    df['Team_B'] = df['Team_B'].astype(str).str.strip().str.title()
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['date', 'home_score', 'away_score'])
    # Sort chronologically to compute rolling ELO correctly
    df = df.sort_values(by='date').reset_index(drop=True)
    
    print(f"Loaded {len(df)} historical international matches. Computing ELO ratings...")
    
    elo_dict = {}
    last_match_date = {} # Trackt Aktivität der Teams, um "Ghost Teams" zu filtern
    
    def get_elo(team):
        if team not in elo_dict:
            elo_dict[team] = 1500.0
        return elo_dict[team]
        
    def get_k_factor(tournament):
        t_lower = str(tournament).lower()
        if 'world cup' in t_lower and 'qualifying' not in t_lower:
            return 60.0  # WC Finals
        elif 'euro' in t_lower or 'copa' in t_lower or 'cup of nations' in t_lower:
            return 40.0  # Continental Finals
        elif 'qualifying' in t_lower or 'nations league' in t_lower:
            return 30.0  # Continental/WC Qualifiers
        return 20.0      # Friendlies
        
    elo_a_list = []
    elo_b_list = []
    
    # Chronological loop
    for idx, row in df.iterrows():
        team_a = row['Team_A']
        team_b = row['Team_B']
        match_date = row['date']
        
        # Speichere das Datum des letzten Matches für jedes Team
        last_match_date[team_a] = match_date
        last_match_date[team_b] = match_date
        
        # 2. Get pre-match ratings
        r_a = get_elo(team_a)
        r_b = get_elo(team_b)
        
        elo_a_list.append(r_a)
        elo_b_list.append(r_b)
        
        score_a = float(row['home_score'])
        score_b = float(row['away_score'])
        
        # Determine match outcome (S_a)
        if score_a > score_b:
            s_a = 1.0  # Win A
        elif score_a == score_b:
            s_a = 0.5  # Draw
        else:
            s_a = 0.0  # Win B
            
        # Apply Home Advantage (H)
        is_neutral = 'neutral_venue' in row and row['neutral_venue'] == True
        tournament_low = str(row['tournament']).lower()
        
        if is_neutral or ('world cup' in tournament_low and 'qualifying' not in tournament_low):
            h_adv = 0.0
        else:
            h_adv = 100.0 # Standard Home Advantage

        # Expected scores
        e_a = 1.0 / (1.0 + 10.0 ** ((r_b - (r_a + h_adv)) / 400.0))
        e_b = 1.0 - e_a
        
        # Goal Difference Index (G)
        gd = abs(score_a - score_b)
        if gd <= 1:
            g_index = 1.0
        elif gd == 2:
            g_index = 1.5
        else:
            g_index = (11.0 + gd) / 8.0
            
        # 3. Anti-Farming Cap: Deckelt den GD-Multiplikator, wenn riesige Favoriten hoch gewinnen.
        # Ein 8:0 von Argentinien gegen Malta soll das System nicht mit künstlichen Punkten aufblasen.
        if (s_a == 1.0 and e_a > 0.80) or (s_a == 0.0 and e_b > 0.80):
            g_index = min(g_index, 1.5)
            
        # Tournament Weight (K)
        k = get_k_factor(row['tournament'])
        
        # Update ratings
        point_exchange = k * g_index * (s_a - e_a)
        
        r_a_new = r_a + point_exchange
        r_b_new = r_b - point_exchange 
        
        # Save back to running dictionary
        elo_dict[team_a] = r_a_new
        elo_dict[team_b] = r_b_new
        
    df['Elo_A'] = elo_a_list
    df['Elo_B'] = elo_b_list
    df['Delta_Elo'] = df['Elo_A'] - df['Elo_B']
    
    # Save processed matches
    processed_dir = os.path.join('data', 'processed', 'features')
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(os.path.join(processed_dir, 'historical_elo_matches.csv'), index=False)
    
    # 4. Ghost-Team Filter: Nur Teams auf die Rangliste setzen, die seit dem 01.01.2021 gespielt haben
    cutoff_date = pd.to_datetime('2021-01-01')
    active_teams = {t for t, d in last_match_date.items() if d >= cutoff_date}
    
    ratings = []
    for team, r in elo_dict.items():
        if team in active_teams:
            ratings.append({
                'Country': team,
                'ELO_Rating': r
            })
            
    elo_ratings_df = pd.DataFrame(ratings).sort_values(by='ELO_Rating', ascending=False)
    elo_ratings_df.to_csv(os.path.join(processed_dir, 'current_elo_ratings.csv'), index=False)
    
    print("-> Mathematical, stabilized ELO calculation complete.")
    print("Top 15 Active National Teams (Cleaned ELO Leaderboard):")
    print(elo_ratings_df.head(15).to_string(index=False))