import pandas as pd
import os

def build_training_data():
    master_path = os.path.join('data', 'processed', 'features', 'MASTER_dataset.csv')
    results_path = os.path.join('data', 'raw', 'historical', 'results.csv')
    
    if not os.path.exists(master_path) or not os.path.exists(results_path):
        print("Fehler: MASTER_dataset.csv oder results.csv fehlt.")
        return
        
    master_df = pd.read_csv(master_path)
    results_df = pd.read_csv(results_path)
    
    print("Baue finale Trainings-Matrix (Deltas + Historische Ergebnisse)...\n" + "-"*40)
    
    # 1. Wir hängen die Features für Team A an
    train_df = pd.merge(results_df, master_df, left_on='Team_A', right_on='Country', how='inner')
    # Spalten umbenennen, damit sie "_A" am Ende haben
    train_df = train_df.rename(columns=lambda x: f"{x}_A" if x in master_df.columns else x)
    
    # 2. Wir hängen die Features für Team B an
    train_df = pd.merge(train_df, master_df, left_on='Team_B', right_on='Country', how='inner')
    # Spalten umbenennen, damit sie "_B" am Ende haben
    train_df = train_df.rename(columns=lambda x: f"{x}_B" if x in master_df.columns else x)
    
    # Durch das 'inner' Joinen sind jetzt nur noch Spiele übrig, 
    # bei denen beide Teams in unseren 5 Test-Nationen sind.
    
    # 3. Deltas berechnen (Team A - Team B)
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
    
    # 4. Nur die relevanten Spalten für das Machine Learning behalten
    features = [
        'Delta_Total_Market_Value', 'Delta_Median_Top11_Value', 'Delta_Chemistry',
        'Delta_Form_Rating', 'Delta_UCL_Minutes', 'Delta_Tournament_Minutes',
        'Delta_Average_Age', 'Delta_TM_Value_Rank', 'Delta_FIFA_Rank', 'Delta_FIFA_Points'
    ]
    
    final_cols = ['date', 'Team_A', 'Team_B', 'home_score', 'away_score', 'target'] + features
    final_train_df = train_df[final_cols].copy()
    
    # 5. Speichern
    out_dir = os.path.join('data', 'processed', 'model_input')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'training_data.csv')
    final_train_df.to_csv(out_path, index=False)
    
    print(f"-> Trainingsdaten erfolgreich erstellt: {len(final_train_df)} elitäre Matches gefunden.")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nVorschau der ML-Trainings-Matrix:")
    print(final_train_df[['date', 'Team_A', 'Team_B', 'target', 'Delta_Total_Market_Value', 'Delta_FIFA_Points']].head(10))

if __name__ == "__main__":
    build_training_data()