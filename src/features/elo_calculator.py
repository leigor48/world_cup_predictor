import os
import numpy as np
import pandas as pd

def calculate_elo_ratings():
    """Chronologically calculates optimized, confederation-weighted ELO ratings to prevent isolated pool inflation."""
    results_path = os.path.join('data', 'raw', 'historical', 'results.csv')
    if not os.path.exists(results_path):
        print(f"Error: Historical results CSV not found at {results_path}. Run scraping step first.")
        return
        
    df = pd.read_csv(results_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort chronologically to compute rolling ELO correctly
    df = df.sort_values(by='date').reset_index(drop=True)
    
    print(f"Loaded {len(df)} historical international matches. Computing ELO ratings...")
    
    # Define continental lists to resolve isolated pool inflation
    afc_teams = {
        'iran', 'japan', 'south korea', 'australia', 'saudi arabia', 'iraq', 'jordan', 'qatar', 
        'uzbekistan', 'china', 'syria', 'vietnam', 'oman', 'india', 'united arab emirates', 
        'lebanon', 'palestine', 'bahrain', 'kuwait', 'hong kong', 'north korea', 'thailand', 
        'tajikistan', 'kyrgyzstan', 'malaysia', 'singapore', 'yemen', 'indonesia', 'afghanistan', 
        'maldives', 'guam', 'taiwan', 'nepal', 'sri lanka', 'cambodia', 'mongolia', 'macau', 
        'laos', 'brunei', 'timor-leste', 'philippines', 'myanmar', 'bangladesh', 'bhutan',
        'korea republic', 'kyrgyz republic', 'yemen pr'
    }
    
    conmebol_teams = {
        'argentina', 'brazil', 'uruguay', 'colombia', 'ecuador', 'chile', 'peru', 'paraguay', 
        'venezuela', 'bolivia'
    }
    
    concacaf_teams = {
        'usa', 'mexico', 'canada', 'panama', 'jamaica', 'costa rica', 'honduras', 'haiti', 
        'curacao', 'trinidad and tobago', 'el salvador', 'guatemala', 'martinique', 'guadeloupe', 
        'suriname', 'bermuda', 'nicaragua', 'grenada', 'cuba', 'barbados', 'dominica', 'aruba', 
        'saint kitts and nevis', 'belize', 'guyana', 'puerto rico', 'st. kitts and nevis', 
        'st. lucia', 'st. vincent and the grenadines'
    }
    
    caf_teams = {
        'morocco', 'senegal', 'egypt', 'algeria', 'tunisia', 'nigeria', 'cameroon', 'ivory coast', 
        'mali', 'dr congo', 'south africa', 'ghana', 'burkina faso', 'cape verde', 'guinea', 
        'gabon', 'zambia', 'uganda', 'angola', 'benin', 'kenya', 'togo', 'libya', 'madagascar', 
        'namibia', 'mauritania', 'guinea-bissau', 'congo', 'equatorial guinea', 'sierra leone', 
        'central african republic', 'rwanda', 'sudan', 'tanzania', 'mozambique', 'malawi', 
        'zimbabwe', 'niger', 'burundi', 'gambia', 'liberia', 'somalia', 'cote d\'ivoire',
        'democratic republic of the congo'
    }
    
    ofc_teams = {
        'new zealand', 'fiji', 'solomon islands', 'tahiti', 'new caledonia', 'vanuatu', 'samoa', 
        'tonga', 'american samoa', 'cook islands', 'papua new guinea', 'new caledonia'
    }

    # Helper to retrieve confederation coefficient
    def get_confederation_coefficient(team):
        t_low = str(team).lower().strip()
        if t_low in afc_teams:
            return 0.75     # AFC (Asia) - highly inflated isolated pool
        elif t_low in conmebol_teams:
            return 1.00     # CONMEBOL (South America) - extremely competitive
        elif t_low in concacaf_teams or t_low in caf_teams:
            return 0.85     # CONCACAF & CAF - moderately competitive
        elif t_low in ofc_teams:
            return 0.70     # OFC (Oceania) - low competition
        return 1.00         # UEFA (Europe) / Other default

    # Initialize all national teams with 1500 baseline ELO
    elo_dict = {}
    
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
        
        # 1. Get pre-match ratings
        r_a = get_elo(team_a)
        r_b = get_elo(team_b)
        
        elo_a_list.append(r_a)
        elo_b_list.append(r_b)
        
        score_a = float(row['home_score'])
        score_b = float(row['away_score'])
        
        # 2. Determine match outcome (S_a)
        if score_a > score_b:
            s_a = 1.0  # Win A
        elif score_a == score_b:
            s_a = 0.5  # Draw
        else:
            s_a = 0.0  # Win B
            
        # 3. Apply Home Advantage (H)
        # Check if match is on neutral ground (if your dataset has this flag)
        # If no flag exists, assume World Cup/Continental Finals are neutral (H=0), qualifiers/friendlies are home (H=100)
        is_neutral = 'neutral_venue' in row and row['neutral_venue'] == True
        tournament_low = str(row['tournament']).lower()
        
        if is_neutral or ('world cup' in tournament_low and 'qualifying' not in tournament_low):
            h_adv = 0.0
        else:
            h_adv = 100.0 # Standard Home Advantage

        # 4. Expected scores with Home Advantage applied to Team A
        e_a = 1.0 / (1.0 + 10.0 ** ((r_b - (r_a + h_adv)) / 400.0))
        e_b = 1.0 - e_a
        
        # 5. Goal Difference Index (G)
        gd = abs(score_a - score_b)
        if gd <= 1:
            g_index = 1.0
        elif gd == 2:
            g_index = 1.5
        else:
            g_index = (11.0 + gd) / 8.0
            
        # 6. Tournament Weight (K)
        k = get_k_factor(row['tournament'])
        
        # 7. Update ratings strictly by the math (no artificial boosters)
        # Formula: R_new = R_old + K * G * (S - E)
        point_exchange = k * g_index * (s_a - e_a)
        
        r_a_new = r_a + point_exchange
        r_b_new = r_b - point_exchange # Elo is a zero-sum game!
        
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
    
    # Save final current ELO leaderboard
    ratings = []
    for team, r in elo_dict.items():
        ratings.append({
            'Country': team,
            'ELO_Rating': r
        })
        
    elo_ratings_df = pd.DataFrame(ratings).sort_values(by='ELO_Rating', ascending=False)
    elo_ratings_df.to_csv(os.path.join(processed_dir, 'current_elo_ratings.csv'), index=False)
    
    print("-> Customized, confederation-weighted ELO calculation complete.")
    print("Top 15 National Teams (Optimized ELO Leaderboard):")
    print(elo_ratings_df.head(15).to_string(index=False))
