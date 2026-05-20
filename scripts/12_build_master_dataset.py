import pandas as pd
import os
import glob

def build_master_dataset():
    features_dir = os.path.join('data', 'processed', 'features')
    squads_dir = os.path.join('data', 'processed', 'squads')
    
    print("Erstelle den Master-Datensatz für alle Nationen...\n" + "-"*40)
    
    # 1. Basis-Werte aus den Squad-Dateien
    squad_features = []
    for file in glob.glob(os.path.join(squads_dir, '*_cleaned.csv')):
        country = os.path.basename(file).replace('_cleaned.csv', '').capitalize()
        # Sonderfälle für Dateinamen (z.B. "south_korea_cleaned.csv")
        country = country.replace('_', ' ')
        
        df = pd.read_csv(file)
        top_11_value = df.nlargest(11, 'Market_Value_mEUR')['Market_Value_mEUR'].median()
        total_value = df['Market_Value_mEUR'].sum()
        
        # Den "Happy Accident" berechnen (Durchschnittsalter)
        average_age = df['Age'].mean() if 'Age' in df.columns else 26.0
        
        squad_features.append({
            'Country': country,
            'Total_Market_Value_mEUR': total_value,
            'Median_Top11_Market_Value_mEUR': top_11_value,
            'Average_Age': average_age
        })
        
    master_df = pd.DataFrame(squad_features)
    
    # 2. Alle Feature-Dateien laden
    feature_files = [
        'club_chemistry.csv',
        'current_form_ratings.csv',
        'ucl_experience.csv',
        'tournament_experience.csv',
        'fifa_ranking.csv' # Unser neues separates Ranking-File!
    ]
    
    for file_name in feature_files:
        file_path = os.path.join(features_dir, file_name)
        if os.path.exists(file_path):
            f_df = pd.read_csv(file_path)
            # Namen-Normalisierung für den Merge
            f_df['Country'] = f_df['Country'].str.title()
            master_df['Country'] = master_df['Country'].str.title()
            
            master_df = pd.merge(master_df, f_df, on='Country', how='left')
            
    # 3. Fehlende Werte füllen (Penaltys für Teams, die z.B. nicht in den Top 75 der FIFA sind)
    master_df['FIFA_Rank'] = master_df['FIFA_Rank'].fillna(100)
    master_df['FIFA_Points'] = master_df['FIFA_Points'].fillna(1000)
    
    # Dummy-Spalte für den TM-Rang, da wir die ja überschrieben hatten
    master_df['TM_Value_Rank'] = master_df['Total_Market_Value_mEUR'].rank(ascending=False)
            
    # 4. Speichern
    master_path = os.path.join(features_dir, 'MASTER_dataset.csv')
    master_df.to_csv(master_path, index=False)
    
    print(f"-> Master-Datensatz erfolgreich für {len(master_df)} Nationen erstellt!")

if __name__ == "__main__":
    build_master_dataset()