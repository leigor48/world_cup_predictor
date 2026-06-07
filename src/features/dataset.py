import os
import glob
import itertools
import pandas as pd

def build_master_dataset():
    """Combines all processed feature files and squad base statistics into a master national dataset."""
    features_dir = os.path.join('data', 'processed', 'features')
    squads_dir = os.path.join('data', 'processed', 'squads')
    
    print("Building master national dataset...")
    squad_features = []
    for file in glob.glob(os.path.join(squads_dir, '*_cleaned.csv')):
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize()
        country = country.replace('_', ' ')
        
        df = pd.read_csv(file)
        top_11_value = df.nlargest(11, 'Market_Value_mEUR')['Market_Value_mEUR'].median()
        total_value = df['Market_Value_mEUR'].sum()
        average_age = df['Age'].mean() if 'Age' in df.columns else 26.0
        
        squad_features.append({
            'Country': country,
            'Total_Market_Value_mEUR': total_value,
            'Median_Top11_Market_Value_mEUR': top_11_value,
            'Average_Age': average_age
        })
        
    master_df = pd.DataFrame(squad_features)
    
    feature_files = [
        'club_chemistry.csv',
        'current_form_ratings.csv',
        'ucl_experience.csv',
        'tournament_experience.csv',
        'fifa_ranking.csv'
    ]
    
    for file_name in feature_files:
        file_path = os.path.join(features_dir, file_name)
        if os.path.exists(file_path):
            f_df = pd.read_csv(file_path)
            f_df['Country'] = f_df['Country'].str.title()
            master_df['Country'] = master_df['Country'].str.title()
            master_df = pd.merge(master_df, f_df, on='Country', how='left')
            
    master_df['FIFA_Rank'] = master_df['FIFA_Rank'].fillna(100)
    master_df['FIFA_Points'] = master_df['FIFA_Points'].fillna(1000)
    master_df['TM_Value_Rank'] = master_df['Total_Market_Value_mEUR'].rank(ascending=False)
            
    master_path = os.path.join(features_dir, 'MASTER_dataset.csv')
    master_df.to_csv(master_path, index=False)
    print(f"-> Master dataset successfully created for {len(master_df)} countries.")


def create_matchups():
    """Generates a complete pairwise matchup feature matrix from the master dataset."""
    master_path = os.path.join('data', 'processed', 'features', 'MASTER_dataset.csv')
    if not os.path.exists(master_path):
        print("Error: MASTER_dataset.csv not found. Build master dataset first.")
        return
        
    df = pd.read_csv(master_path)
    print("Generating matchup matrices...")
    
    matchups = []
    teams = df['Country'].tolist()
    combinations = list(itertools.combinations(teams, 2))
    
    for team_a, team_b in combinations:
        data_a = df[df['Country'] == team_a].iloc[0]
        data_b = df[df['Country'] == team_b].iloc[0]
        
        matchup_data = {
            'Team_A': team_a,
            'Team_B': team_b,
            'Delta_Total_Market_Value': data_a['Total_Market_Value_mEUR'] - data_b['Total_Market_Value_mEUR'],
            'Delta_Median_Top11_Value': data_a['Median_Top11_Market_Value_mEUR'] - data_b['Median_Top11_Market_Value_mEUR'],
            'Delta_Chemistry': data_a['Chemistry_Score'] - data_b['Chemistry_Score'],
            'Delta_Form_Rating': data_a['Current_Form_Rating'] - data_b['Current_Form_Rating'],
            'Delta_UCL_Minutes': data_a['Total_UCL_Minutes'] - data_b['Total_UCL_Minutes'],
            'Delta_Tournament_Minutes': data_a['Total_Tournament_Minutes'] - data_b['Total_Tournament_Minutes'],
            'Delta_Average_Age': data_a['Average_Age'] - data_b['Average_Age'],
            'Delta_TM_Value_Rank': data_a['TM_Value_Rank'] - data_b['TM_Value_Rank'],
            'Delta_FIFA_Rank': data_a['FIFA_Rank'] - data_b['FIFA_Rank'],
            'Delta_FIFA_Points': data_a['FIFA_Points'] - data_b['FIFA_Points']
        }
        matchups.append(matchup_data)
        
    matchup_df = pd.DataFrame(matchups)
    out_dir = os.path.join('data', 'processed', 'model_input')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'matchup_features.csv')
    matchup_df.to_csv(out_path, index=False)
    print(f"-> Pairwise matchups successfully created ({len(combinations)} pairs).")


def build_training_data():
    """Generates the final multi-feature training dataset by combining historical matches with team deltas."""
    master_path = os.path.join('data', 'processed', 'features', 'MASTER_dataset.csv')
    results_path = os.path.join('data', 'raw', 'historical', 'results.csv')
    
    if not os.path.exists(master_path) or not os.path.exists(results_path):
        print("Error: Missing master dataset or historical results files.")
        return
        
    master_df = pd.read_csv(master_path)
    results_df = pd.read_csv(results_path)
    
    print("Building final training dataset matrix...")
    
    train_df = pd.merge(results_df, master_df, left_on='Team_A', right_on='Country', how='inner')
    train_df = train_df.rename(columns=lambda x: f"{x}_A" if x in master_df.columns else x)
    
    train_df = pd.merge(train_df, master_df, left_on='Team_B', right_on='Country', how='inner')
    train_df = train_df.rename(columns=lambda x: f"{x}_B" if x in master_df.columns else x)
    
    train_df['Delta_Total_Market_Value'] = train_df['Total_Market_Value_mEUR_A'] - train_df['Total_Market_Value_mEUR_B']
    train_df['Delta_Median_Top11_Value'] = train_df['Median_Top11_Market_Value_mEUR_A'] - train_df['Median_Top11_Market_Value_mEUR_B']
    train_df['Delta_Chemistry'] = train_df['Chemistry_Score_A'] - train_df['Chemistry_Score_B']
    train_df['Delta_Form_Rating'] = train_df['Current_Form_Rating_A'] - train_df['Current_Form_Rating_B']
    train_df['Delta_UCL_Minutes'] = train_df['Total_UCL_Minutes_A'] - train_df['Total_UCL_Minutes_B']
    train_df['Delta_Tournament_Minutes'] = train_df['Total_Tournament_Minutes_A'] - train_df['Total_Tournament_Minutes_B']
    train_df['Delta_Average_Age'] = train_df['Average_Age_A'] - train_df['Average_Age_B']
    train_df['Delta_TM_Value_Rank'] = train_df['TM_Value_Rank_A'] - train_df['TM_Value_Rank_B']
    train_df['Delta_FIFA_Rank'] = train_df['FIFA_Rank_A'] - train_df['FIFA_Rank_B']
    train_df['Delta_FIFA_Points'] = train_df['FIFA_Points_A'] - train_df['FIFA_Points_B']
    
    train_df['Is_Neutral'] = train_df['neutral'].astype(int)
    
    features = [
        'Delta_Total_Market_Value', 'Delta_Median_Top11_Value', 'Delta_Chemistry',
        'Delta_Form_Rating', 'Delta_UCL_Minutes', 'Delta_Tournament_Minutes',
        'Delta_Average_Age', 'Delta_TM_Value_Rank', 'Delta_FIFA_Rank', 'Delta_FIFA_Points',
        'Is_Neutral'
    ]
    
    final_cols = ['date', 'Team_A', 'Team_B', 'home_score', 'away_score', 'target'] + features
    final_train_df = train_df[final_cols].copy()
    
    out_dir = os.path.join('data', 'processed', 'model_input')
    os.makedirs(out_dir, exist_ok=True)
    final_train_df.to_csv(os.path.join(out_dir, 'training_data.csv'), index=False)
    print("-> Training dataset matrix creation complete.")
