#!/usr/bin/env python3
"""
Generate realistic NHL game data and train XGBoost model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
from datetime import datetime, timedelta

np.random.seed(42)

TEAMS = [
    'ANA', 'BOS', 'BUF', 'CAR', 'CBJ', 'CGY', 'CHI', 'COL', 'DAL', 'DET',
    'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NJD', 'NSH', 'NYI', 'NYR', 'OTT',
    'PHI', 'PIT', 'SJS', 'SEA', 'STL', 'TBL', 'TOR', 'UTA', 'VAN', 'VGK', 'WPG', 'WSH'
]

TEAM_NAMES = {
    'ANA': 'Anaheim Ducks', 'BOS': 'Boston Bruins', 'BUF': 'Buffalo Sabres',
    'CAR': 'Carolina Hurricanes', 'CBJ': 'Columbus Blue Jackets', 'CGY': 'Calgary Flames',
    'CHI': 'Chicago Blackhawks', 'COL': 'Colorado Avalanche', 'DAL': 'Dallas Stars',
    'DET': 'Detroit Red Wings', 'EDM': 'Edmonton Oilers', 'FLA': 'Florida Panthers',
    'LAK': 'Los Angeles Kings', 'MIN': 'Minnesota Wild', 'MTL': 'Montreal Canadiens',
    'NJD': 'New Jersey Devils', 'NSH': 'Nashville Predators', 'NYI': 'New York Islanders',
    'NYR': 'New York Rangers', 'OTT': 'Ottawa Senators', 'PHI': 'Philadelphia Flyers',
    'PIT': 'Pittsburgh Penguins', 'SJS': 'San Jose Sharks', 'SEA': 'Seattle Kraken',
    'STL': 'St Louis Blues', 'TBL': 'Tampa Bay Lightning', 'TOR': 'Toronto Maple Leafs',
    'UTA': 'Utah Hockey Club', 'VAN': 'Vancouver Canucks', 'VGK': 'Vegas Golden Knights',
    'WPG': 'Winnipeg Jets', 'WSH': 'Washington Capitals'
}

SEASONS = [
    ('20132014', '2013-10-01', '2014-04-14'),
    ('20142015', '2014-10-01', '2015-04-11'),
    ('20152016', '2015-10-01', '2016-04-10'),
    ('20162017', '2016-10-12', '2017-04-09'),
    ('20172018', '2017-10-04', '2018-04-08'),
    ('20182019', '2018-10-03', '2019-04-06'),
    ('20192020', '2019-10-02', '2020-03-12'),
    ('20202021', '2021-01-13', '2021-05-16'),
    ('20212022', '2021-10-12', '2022-04-29'),
    ('20222023', '2022-10-11', '2023-04-16'),
    ('20232024', '2023-10-10', '2024-04-18'),
    ('20242025', '2024-10-08', '2025-04-17'),
    ('20252026', '2025-10-07', '2026-04-15'),
]

def generate_team_stats(team: str, season: str) -> dict:
    """Generate realistic team stats based on typical NHL patterns"""
    base_cf_pct = np.random.uniform(40, 60)
    base_xgf_pct = np.random.uniform(40, 60)
    base_hdcf_pct = np.random.uniform(35, 65)
    
    team_factor = hash(team) % 20 - 10
    season_factor = hash(season) % 10 - 5
    
    return {
        'cf_pct': max(35, min(65, base_cf_pct + team_factor + season_factor)),
        'xgf_pct': max(35, min(65, base_xgf_pct + team_factor + season_factor)),
        'hdcf_pct': max(25, min(75, base_hdcf_pct + team_factor + season_factor)),
        'sf_pct': max(40, min(60, base_cf_pct * 0.9 + team_factor + season_factor)),
        'gf_pct': max(35, min(65, base_xgf_pct * 1.1 + team_factor + season_factor + np.random.uniform(-5, 5))),
    }

def generate_games() -> pd.DataFrame:
    """Generate realistic NHL game data"""
    games = []
    
    for season_code, start_date, end_date in SEASONS:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current = start
        game_id = 0
        
        while current <= end:
            if current.weekday() < 5 and np.random.random() < 0.3:
                game_id += 1
                
                home_team = np.random.choice(TEAMS)
                away_team = np.random.choice([t for t in TEAMS if t != home_team])
                
                home_stats = generate_team_stats(home_team, season_code)
                away_stats = generate_team_stats(away_team, season_code)
                
                home_advantage = np.random.uniform(2, 5)
                
                home_expected_goals = max(0.5, (home_stats['xgf_pct'] / 50 + away_stats['xgf_pct'] / 50 - 1) * 1.5 + 2.5 + home_advantage / 10)
                away_expected_goals = max(0.5, 3.0 - home_expected_goals + np.random.uniform(-0.5, 0.5))
                
                home_goals = max(0, int(np.random.poisson(home_expected_goals)))
                away_goals = max(0, int(np.random.poisson(away_expected_goals)))
                
                games.append({
                    'season': season_code,
                    'date': current.strftime('%Y-%m-%d'),
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_gf': home_goals,
                    'away_gf': away_goals,
                    'home_cf_pct': home_stats['cf_pct'],
                    'away_cf_pct': away_stats['cf_pct'],
                    'home_xgf_pct': home_stats['xgf_pct'],
                    'away_xgf_pct': away_stats['xgf_pct'],
                    'home_hdcf_pct': home_stats['hdcf_pct'],
                    'away_hdcf_pct': away_stats['hdcf_pct'],
                    'home_sf_pct': home_stats['sf_pct'],
                    'away_sf_pct': away_stats['sf_pct'],
                    'home_gf_pct': home_stats['gf_pct'],
                    'away_gf_pct': away_stats['gf_pct'],
                    'home_goals_2_5': 1 if home_goals >= 3 else 0,
                })
            
            current += timedelta(days=1)
    
    return pd.DataFrame(games)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for model training"""
    features = pd.DataFrame()
    
    features['home_cf_pct'] = df['home_cf_pct']
    features['away_cf_pct'] = df['away_cf_pct']
    features['cf_diff'] = df['home_cf_pct'] - df['away_cf_pct']
    
    features['home_xgf_pct'] = df['home_xgf_pct']
    features['away_xgf_pct'] = df['away_xgf_pct']
    features['xgf_diff'] = df['home_xgf_pct'] - df['away_xgf_pct']
    
    features['home_hdcf_pct'] = df['home_hdcf_pct']
    features['away_hdcf_pct'] = df['away_hdcf_pct']
    features['hdcf_diff'] = df['home_hdcf_pct'] - df['away_hdcf_pct']
    
    features['home_sf_pct'] = df['home_sf_pct']
    features['away_sf_pct'] = df['away_sf_pct']
    features['sf_diff'] = df['home_sf_pct'] - df['away_sf_pct']
    
    features['home_gf_pct'] = df['home_gf_pct']
    features['away_gf_pct'] = df['away_gf_pct']
    features['gf_diff'] = df['home_gf_pct'] - df['away_gf_pct']
    
    features['home_advantage'] = 1
    
    return features

def extract_season_year(season_code: str) -> int:
    """Extract year from season code"""
    return int(season_code[:4])

def main():
    print("Generating NHL game data...")
    df = generate_games()
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/nhl_games.csv', index=False)
    print(f"Generated {len(df)} games")
    
    print("\nPreparing features...")
    X = prepare_features(df)
    y = df['home_goals_2_5']
    
    df['season_year'] = df['season'].apply(extract_season_year)
    
    train_mask = df['season_year'] <= 2022
    test_mask = df['season_year'] > 2022
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"Training set: {len(X_train)} games")
    print(f"Test set: {len(X_test)} games")
    print(f"Target distribution (train): {y_train.mean():.2%} with 2+ goals")
    
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['<2 goals', '2+ goals']))
    
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/hockey_model.pkl')
    print("\nModel saved to model/hockey_model.pkl")
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))
    
    test_results = df[test_mask].copy()
    test_results['prediction'] = y_pred_proba
    test_results.to_csv('data/test_results.csv', index=False)
    print("\nTest results saved to data/test_results.csv")

if __name__ == '__main__':
    main()
