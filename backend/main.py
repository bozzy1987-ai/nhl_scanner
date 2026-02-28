#!/usr/bin/env python3
"""
FastAPI backend for NHL Hockey Analytics - dynamic training
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
import json

DATA_PATH = Path(__file__).parent / "data" / "nhl_game_results.csv"
TEAM_STATS_PATH = Path(__file__).parent / "data" / "teams_2008_to_2024.csv"
DATA_2024_25_PATH = Path(__file__).parent / "data" / "test_predictions_2024_25.csv"
DATA_2025_26_PATH = Path(__file__).parent / "data" / "nhl_game_results_2025_26.csv"
MODELS_PATH = Path(__file__).parent / "models"

feature_cols = ['home_xg_pct', 'home_cf_pct', 'away_xg_pct', 'away_cf_pct',
                'home_goals_for', 'home_goals_against', 'away_goals_for', 'away_goals_against']

feature_cols_v2 = ['home_xg_pct', 'home_cf_pct', 'home_ff_pct', 'away_xg_pct', 'away_cf_pct', 'away_ff_pct',
                   'home_goals_for', 'home_goals_against', 'away_goals_for', 'away_goals_against',
                   'home_shots_for', 'home_shots_against', 'away_shots_for', 'away_shots_against',
                   'home_high_danger', 'away_high_danger']

df = None
df_2024_25 = None
df_2025_26 = None
combined_df = None
teams = None

def load_data():
    global df, df_2024_25, df_2025_26, combined_df, teams
    try:
        # Load team stats
        teams = pd.read_csv(TEAM_STATS_PATH)
        teams = teams[teams['situation'] == 'all']
        
        # Load main game results (seasons 2013-2024 + 2025-26)
        games = pd.read_csv(DATA_PATH)
        games['season'] = games['season'].astype(int)
        
        # Fix team names
        games['home_team'] = games['home_team'].replace('PHX', 'ARI')
        games['away_team'] = games['away_team'].replace('PHX', 'ARI')
        
        # Build features for main games
        features = []
        games['season_year'] = games['season'].astype(str).str[:4].astype(int)
        
        for idx, row in games.iterrows():
            season_year = row['season_year']
            home = row['home_team']
            away = row['away_team']
            
            home_stats = teams[(teams['team'] == home) & (teams['season'] == season_year)]
            away_stats = teams[(teams['team'] == away) & (teams['season'] == season_year)]
            
            if len(home_stats) == 0 or len(away_stats) == 0:
                continue
            
            home_s = home_stats.iloc[0]
            away_s = away_stats.iloc[0]
            
            # For v2: fallback to 2023 data if current season has NaN
            if season_year == 2024:
                home_2023 = teams[(teams['team'] == home) & (teams['season'] == 2023)]
                away_2023 = teams[(teams['team'] == away) & (teams['season'] == 2023)]
                if len(home_2023) > 0:
                    home_2023 = home_2023.iloc[0]
                    if pd.isna(home_s.get('fenwickPercentage')) or home_s.get('fenwickPercentage') == 0:
                        home_s['fenwickPercentage'] = home_2023.get('fenwickPercentage', 50)
                    if pd.isna(home_s.get('shotsOnGoalFor')) or home_s.get('shotsOnGoalFor') == 0:
                        home_s['shotsOnGoalFor'] = home_2023.get('shotsOnGoalFor', 0)
                    if pd.isna(home_s.get('shotsOnGoalAgainst')) or home_s.get('shotsOnGoalAgainst') == 0:
                        home_s['shotsOnGoalAgainst'] = home_2023.get('shotsOnGoalAgainst', 0)
                    if pd.isna(home_s.get('highDangerShotsFor')) or home_s.get('highDangerShotsFor') == 0:
                        home_s['highDangerShotsFor'] = home_2023.get('highDangerShotsFor', 0)
                if len(away_2023) > 0:
                    away_2023 = away_2023.iloc[0]
                    if pd.isna(away_s.get('fenwickPercentage')) or away_s.get('fenwickPercentage') == 0:
                        away_s['fenwickPercentage'] = away_2023.get('fenwickPercentage', 50)
                    if pd.isna(away_s.get('shotsOnGoalFor')) or away_s.get('shotsOnGoalFor') == 0:
                        away_s['shotsOnGoalFor'] = away_2023.get('shotsOnGoalFor', 0)
                    if pd.isna(away_s.get('shotsOnGoalAgainst')) or away_s.get('shotsOnGoalAgainst') == 0:
                        away_s['shotsOnGoalAgainst'] = away_2023.get('shotsOnGoalAgainst', 0)
                    if pd.isna(away_s.get('highDangerShotsFor')) or away_s.get('highDangerShotsFor') == 0:
                        away_s['highDangerShotsFor'] = away_2023.get('highDangerShotsFor', 0)
            
            features.append({
                'season': row['season'],
                'date': row['date'],
                'home_team': home,
                'away_team': away,
                'home_gf': row['home_gf'],
                'away_gf': row['away_gf'],
                'home_xg_pct': home_s['xGoalsPercentage'],
                'home_cf_pct': home_s['corsiPercentage'],
                'home_ff_pct': home_s.get('fenwickPercentage', 50),
                'away_xg_pct': away_s['xGoalsPercentage'],
                'away_cf_pct': away_s['corsiPercentage'],
                'away_ff_pct': away_s.get('fenwickPercentage', 50),
                'home_goals_for': home_s['goalsFor'],
                'home_goals_against': home_s['goalsAgainst'],
                'away_goals_for': away_s['goalsFor'],
                'away_goals_against': away_s['goalsAgainst'],
                'home_shots_for': home_s.get('shotsOnGoalFor', 0),
                'home_shots_against': home_s.get('shotsOnGoalAgainst', 0),
                'away_shots_for': away_s.get('shotsOnGoalFor', 0),
                'away_shots_against': away_s.get('shotsOnGoalAgainst', 0),
                'home_high_danger': home_s.get('highDangerShotsFor', 0),
                'away_high_danger': away_s.get('highDangerShotsFor', 0),
            })
        
        df = pd.DataFrame(features)
        print(f"Loaded {len(df)} games (2013-2024 + 2025-26)")
        
        # Add v2 columns to df
        if 'home_ff_pct' not in df.columns:
            df['home_ff_pct'] = 50.0
            df['away_ff_pct'] = 50.0
            df['home_shots_for'] = 0
            df['home_shots_against'] = 0
            df['away_shots_for'] = 0
            df['away_shots_against'] = 0
            df['home_high_danger'] = 0
            df['away_high_danger'] = 0
        
        # Load 2024-25 data if available
        if DATA_2024_25_PATH.exists():
            df_2024_25 = pd.read_csv(DATA_2024_25_PATH)
            df_2024_25['season'] = 20242025
            df_2024_25['home_team'] = df_2024_25['home_team'].replace('UTA', 'ARI')
            df_2024_25['away_team'] = df_2024_25['away_team'].replace('UTA', 'ARI')
            # Add v2 columns
            df_2024_25['home_ff_pct'] = 50.0
            df_2024_25['away_ff_pct'] = 50.0
            df_2024_25['home_shots_for'] = 0
            df_2024_25['home_shots_against'] = 0
            df_2024_25['away_shots_for'] = 0
            df_2024_25['away_shots_against'] = 0
            df_2024_25['home_high_danger'] = 0
            df_2024_25['away_high_danger'] = 0
            print(f"Loaded {len(df_2024_25)} 2024-25 games")
        
        # Load 2025-26 data if available
        if DATA_2025_26_PATH.exists():
            df_2025_26 = pd.read_csv(DATA_2025_26_PATH)
            df_2025_26['season'] = 20252026
            df_2025_26['home_team'] = df_2025_26['home_team'].replace('UTA', 'ARI')
            df_2025_26['away_team'] = df_2025_26['away_team'].replace('UTA', 'ARI')
            # Add v2 columns
            df_2025_26['home_ff_pct'] = 50.0
            df_2025_26['away_ff_pct'] = 50.0
            df_2025_26['home_shots_for'] = 0
            df_2025_26['home_shots_against'] = 0
            df_2025_26['away_shots_for'] = 0
            df_2025_26['away_shots_against'] = 0
            df_2025_26['home_high_danger'] = 0
            df_2025_26['away_high_danger'] = 0
            print(f"Loaded {len(df_2025_26)} 2025-26 games")
        
        # Load new games if exists (auto-updated)
        NEW_GAMES_PATH = Path(__file__).parent / "data" / "new_games_2025_26.csv"
        if NEW_GAMES_PATH.exists():
            df_new = pd.read_csv(NEW_GAMES_PATH)
            df_new['season'] = 20252026
            if df_2025_26 is None:
                df_2025_26 = df_new
            else:
                df_2025_26 = pd.concat([df_2025_26, df_new], ignore_index=True)
                df_2025_26 = df_2025_26.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='last')
            print(f"Loaded {len(df_new)} new games")
        
        # Create combined_df for training
        all_data = [df]
        if df_2024_25 is not None:
            all_data.append(df_2024_25)
        if df_2025_26 is not None:
            all_data.append(df_2025_26)
        combined_df = pd.concat(all_data, ignore_index=True)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    MODELS_PATH.mkdir(exist_ok=True)
    load_data()
    yield

app = FastAPI(title="Hockey Analytics API", version="3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    train_season_start: int
    train_season_end: int
    test_season_start: int
    test_season_end: int
    confidence_threshold: float
    bet_amount: float
    model_version: str = "v1"

@app.get("/")
async def root():
    return {"message": "Hockey Analytics API v3.0 - Dynamic Training"}

@app.get("/seasons")
async def get_seasons():
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    seasons = sorted(df['season'].unique())
    result = [int(str(s)[:4]) for s in seasons]
    
    # Add 2024 for 2024-25, 2025 for 2025-26
    if df_2024_25 is not None:
        result.append(2024)
    if df_2025_26 is not None:
        result.append(2025)
    
    return {"seasons": sorted(set(result))}

@app.post("/simulate")
async def simulate(request: SimulationRequest):
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Convert years to season codes
    def year_to_season_code(year):
        return year * 10000 + (year + 1) % 10000
    
    train_seasons = [year_to_season_code(y) for y in range(request.train_season_start, request.train_season_end + 1)]
    test_seasons = [year_to_season_code(y) for y in range(request.test_season_start, request.test_season_end + 1)]
    
    # Combine all available data for training
    all_data = [df]
    if df_2024_25 is not None:
        all_data.append(df_2024_25)
    if df_2025_26 is not None:
        all_data.append(df_2025_26)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Filter training data
    train_df = combined_df[combined_df['season'].isin(train_seasons)]
    
    # Handle test data - check if 2024-25 or 2025-26 is requested
    if 20242025 in test_seasons and df_2024_25 is not None:
        test_df = df_2024_25.copy()
    elif 20252026 in test_seasons and df_2025_26 is not None:
        test_df = df_2025_26.copy()
    else:
        test_df = combined_df[combined_df['season'].isin(test_seasons)].copy()
    
    if len(train_df) == 0:
        raise HTTPException(status_code=400, detail="Brak danych treningowych")
    if len(test_df) == 0:
        raise HTTPException(status_code=400, detail="Brak danych testowych")
    
    # Create target
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['home_3_plus'] = (train_df['home_gf'] >= 3).astype(int)
    test_df['home_3_plus'] = (test_df['home_gf'] >= 3).astype(int)
    
    # Select features based on model version
    features = feature_cols_v2 if request.model_version.startswith("v2") else feature_cols
    
    # Train model dynamically - always train, don't use saved model for backtesting
    X_train = train_df[features]
    y_train = train_df['home_3_plus']
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Predict
    X_test = test_df[features]
    probabilities = model.predict_proba(X_test)[:, 1]
    test_df['predicted_prob'] = probabilities
    
    # Filter by threshold
    threshold = request.confidence_threshold / 100
    qualifying_bets = test_df[test_df['predicted_prob'] >= threshold].copy()
    
    if len(qualifying_bets) == 0:
        return {
            "total_profit": 0.0,
            "roi_percent": 0.0,
            "hit_rate": 0.0,
            "total_matches": len(test_df),
            "matched_bets": 0,
            "bankroll_history": [],
            "bets": [],
            "comment": "Brak meczów spełniających próg confidence."
        }
    
    # Calculate results
    qualifying_bets['won'] = (qualifying_bets['home_3_plus'] == 1).astype(int)
    qualifying_bets['profit'] = np.where(
        qualifying_bets['won'] == 1,
        request.bet_amount * 0.5,
        -request.bet_amount
    )
    
    # Build history
    bankroll = 0.0
    bankroll_history = []
    for idx, row in qualifying_bets.iterrows():
        bankroll += row['profit']
        bankroll_history.append({
            "match": f"{row['home_team']} vs {row['away_team']}",
            "date": str(row['date']),
            "predicted_prob": round(row['predicted_prob'] * 100, 1),
            "result": "WIN" if row['won'] == 1 else "LOSS",
            "profit": round(row['profit'], 2),
            "bankroll": round(bankroll, 2),
            "score": f"{row['home_gf']}-{row['away_gf']}"
        })
    
    total_profit = bankroll
    total_staked = len(qualifying_bets) * request.bet_amount
    roi_percent = (total_profit / total_staked * 100) if total_staked > 0 else 0
    hit_rate = qualifying_bets['won'].mean() * 100 if len(qualifying_bets) > 0 else 0
    
    return {
        "total_profit": round(total_profit, 2),
        "roi_percent": round(roi_percent, 2),
        "hit_rate": round(hit_rate, 2),
        "total_matches": len(test_df),
        "matched_bets": len(qualifying_bets),
        "bankroll_history": bankroll_history,
        "bets": bankroll_history,
        "comment": f"Model trenowany na {len(X_train)} meczach. ROI: {roi_percent:.1f}%, Hit rate: {hit_rate:.1f}%"
    }

@app.get("/schedule")
async def get_schedule(days_ahead: int = 10, threshold: float = 80.0, model_version: str = "v1"):
    """Get upcoming games and predictions"""
    import requests
    from datetime import datetime, timedelta
    
    # Auto-update: fetch recent results from last 7 days and add to training data
    global combined_df, df
    try:
        recent_games = []
        for day_offset in range(1, 8):  # last 7 days
            date = datetime.now() - timedelta(days=day_offset)
            date_str = date.strftime('%Y-%m-%d')
            
            try:
                resp = requests.get(f"https://api-web.nhle.com/v1/schedule/{date_str}", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('gameWeek'):
                        for day in data['gameWeek']:
                            for game in day.get('games', []):
                                home = game.get('homeTeam', {}).get('abbrev', '')
                                away = game.get('awayTeam', {}).get('abbrev', '')
                                home_score = game.get('homeTeam', {}).get('score', 0)
                                away_score = game.get('awayTeam', {}).get('score', 0)
                                # Only add finished games (score > 0)
                                if home and away and home_score > 0 and away_score > 0:
                                    game_date = day.get('date', date_str)
                                    recent_games.append({
                                        'date': game_date,
                                        'home_team': home,
                                        'away_team': away,
                                        'home_gf': home_score,
                                        'away_gf': away_score,
                                        'season': 20252026
                                    })
            except:
                pass
        
        if recent_games and combined_df is not None:
            # Add recent games to combined_df if not already present
            recent_df = pd.DataFrame(recent_games)
            
            # Map team names
            team_map = {'PHX': 'ARI', 'UTA': 'ARI'}
            recent_df['home_team'] = recent_df['home_team'].replace(team_map)
            recent_df['away_team'] = recent_df['away_team'].replace(team_map)
            
            # Get team stats for 2024-25
            teams_2024 = teams[teams['season'] == 2024].copy()
            
            # Build features for recent games
            new_features = []
            for idx, row in recent_df.iterrows():
                homeMapped = team_map.get(row['home_team'], row['home_team'])
                awayMapped = team_map.get(row['away_team'], row['away_team'])
                
                home_stats = teams_2024[teams_2024['team'] == homeMapped]
                away_stats = teams_2024[teams_2024['team'] == awayMapped]
                
                if len(home_stats) > 0 and len(away_stats) > 0:
                    home_s = home_stats.iloc[0]
                    away_s = away_stats.iloc[0]
                    
                    new_features.append({
                        'season': row['season'],
                        'date': row['date'],
                        'home_team': row['home_team'],
                        'away_team': row['away_team'],
                        'home_gf': row['home_gf'],
                        'away_gf': row['away_gf'],
                        'home_xg_pct': home_s['xGoalsPercentage'],
                        'home_cf_pct': home_s['corsiPercentage'],
                        'away_xg_pct': away_s['xGoalsPercentage'],
                        'away_cf_pct': away_s['corsiPercentage'],
                        'home_goals_for': home_s['goalsFor'],
                        'home_goals_against': home_s['goalsAgainst'],
                        'away_goals_for': away_s['goalsFor'],
                        'away_goals_against': away_s['goalsAgainst'],
                    })
            
            if new_features:
                new_games_df = pd.DataFrame(new_features)
                # Remove duplicates based on date and teams
                combined_df = pd.concat([combined_df, new_games_df], ignore_index=True)
                # Remove duplicates
                combined_df = combined_df.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='last')
                print(f"Added {len(new_games_df)} new games to training data. Total: {len(combined_df)}")
                
                # Save new games to separate file for persistence
                try:
                    new_file = Path(__file__).parent / "data" / "new_games_2025_26.csv"
                    new_games_df.to_csv(new_file, mode='a', header=not new_file.exists(), index=False)
                    print(f"Saved {len(new_games_df)} new games to {new_file}")
                except Exception as save_err:
                    print(f"Save error: {save_err}")
    except Exception as e:
        print(f"Auto-update error: {e}")
    
    try:
        # Fetch games from NHL API
        all_games = []
        now = datetime.now()
        
        # Check if v1_v2 mode (apply B2B filter)
        apply_b2b_filter = (model_version == "v1_v2")
        
        # Get teams that played yesterday (B2B) - for all models
        b2b_teams = set()
        # Check yesterday only (day_offset = 1)
        date = now - timedelta(days=1)
        date_str = date.strftime('%Y-%m-%d')
        
        try:
            resp = requests.get(f'https://api-web.nhle.com/v1/schedule/{date_str}', timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('gameWeek'):
                    for day in data['gameWeek']:
                        day_date = day.get('date', '')
                        if day_date != date_str:
                            continue
                        for game in day.get('games', []):
                            if game.get('gameState') == 'OFF':
                                home = game.get('homeTeam', {}).get('abbrev', '')
                                away = game.get('awayTeam', {}).get('abbrev', '')
                                if home:
                                    b2b_teams.add(home)
                                if away:
                                    b2b_teams.add(away)
        except:
            pass
        
        # Fetch upcoming games
        for day_offset in range(days_ahead):
            date = now + timedelta(days=day_offset)
            date_str = date.strftime('%Y-%m-%d')
            
            try:
                resp = requests.get(f"https://api-web.nhle.com/v1/schedule/{date_str}", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('gameWeek') and len(data.get('gameWeek', [])) > 0:
                        for day in data['gameWeek']:
                            game_date = day.get('date', date_str)
                            for game in day.get('games', []):
                                home = game.get('homeTeam', {}).get('abbrev', '')
                                away = game.get('awayTeam', {}).get('abbrev', '')
                                if home and away:
                                    # Apply B2B filter only for v1_v2 mode and only for TOMORROW+ (not today)
                                    if apply_b2b_filter and day_offset > 0:
                                        if home in b2b_teams or away in b2b_teams:
                                            continue
                                    
                                    all_games.append({
                                        'date': game_date,
                                        'home_team': home,
                                        'away_team': away,
                                    })
            except:
                pass
        
        # Remove duplicates based on date and teams
        seen = set()
        unique_games = []
        for game in all_games:
            key = (game['date'], game['home_team'], game['away_team'])
            if key not in seen:
                seen.add(key)
                unique_games.append(game)
        all_games = unique_games
        
        if not all_games:
            return {"games": [], "message": "No upcoming games found"}
        
        # Return raw schedule if data not loaded
        if df is None:
            return {"games": all_games[:50], "message": f"Found {len(all_games)} games"}
        
        # Build predictions
        return await _build_predictions(all_games, threshold, model_version, b2b_teams)
        
    except Exception as e:
        return {"error": str(e)[:200], "games": []}


async def _build_predictions(all_games, threshold, model_version="v1", b2b_teams=None):
    """Helper to build ML predictions"""
    global combined_df
    
    if b2b_teams is None:
        b2b_teams = set()
    
    teams_2024_25 = teams[teams['season'] == 2024].copy()
    team_map = {'PHX': 'ARI', 'UTA': 'ARI'}
    
    # Build combined_df if not exists
    if combined_df is None:
        all_data = [df]
        if df_2024_25 is not None:
            all_data.append(df_2024_25)
        if df_2025_26 is not None:
            all_data.append(df_2025_26)
        combined_df = pd.concat(all_data, ignore_index=True)
    
    combined_df_ref = combined_df
    
    if len(teams_2024_25) == 0:
        return {"games": all_games[:50], "message": "No team stats"}
    
    # Check if v1_v2 mode (both models must agree)
    use_both = model_version == "v1_v2"
    use_both_low = model_version == "v1_v2_low"
    use_both_mid = model_version == "v1_v2_mid"
    mode_msg = ""
    
    # Build features
    predictions = []
    for game in all_games:
        homeMapped = team_map.get(game['home_team'], game['home_team'])
        awayMapped = team_map.get(game['away_team'], game['away_team'])
        
        home_stats = teams_2024_25[teams_2024_25['team'] == homeMapped]
        away_stats = teams_2024_25[teams_2024_25['team'] == awayMapped]
        
        if len(home_stats) == 0 or len(away_stats) == 0:
            continue
        
        home_s = home_stats.iloc[0]
        away_s = away_stats.iloc[0]
        
        predictions.append({
            'date': game['date'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'home_xg_pct': float(home_s['xGoalsPercentage']),
            'away_xg_pct': float(away_s['xGoalsPercentage']),
        })
    
    # Train model - same as backtest: 2013-2014 to 2024-2025
    train_seasons = [20132014, 20142015, 20152016, 20162017, 20172018, 20182019, 20192020, 20202021, 20212022, 20222023, 20232024, 20242025]
    train_df = combined_df_ref[combined_df_ref['season'].isin(train_seasons)].copy()
    train_df['home_3_plus'] = (train_df['home_gf'] >= 3).astype(int)
    
    # Always train both models for v1_v2 mode
    if use_both:
        features = feature_cols  # V1 features
    else:
        features = feature_cols_v2 if model_version.startswith("v2") else feature_cols
    
    X_train = train_df[features]
    y_train = train_df['home_3_plus']
    
    # Train V1 model
    model_v1 = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
    )
    model_v1.fit(X_train, y_train)
    
    # Train V2 model if needed
    model_v2 = None
    if use_both or use_both_mid or use_both_low or model_version.startswith("v2"):
        features_v2 = feature_cols_v2
        X_train_v2 = train_df[features_v2]
        model_v2 = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
        )
        model_v2.fit(X_train_v2, y_train)
    
    # Predict
    threshold_val = threshold / 100
    results = []
    bet_count = 0
    
    use_v2 = model_version.startswith("v2") or use_both or use_both_mid or use_both_low
    
    for pred in predictions:
        homeMapped = team_map.get(pred['home_team'], pred['home_team'])
        awayMapped = team_map.get(pred['away_team'], pred['away_team'])
        
        home_s = teams_2024_25[teams_2024_25['team'] == homeMapped].iloc[0]
        away_s = teams_2024_25[teams_2024_25['team'] == awayMapped].iloc[0]
        
        # For v2: fallback to 2023 data if 2024 has NaN
        if use_v2:
            teams_2023 = teams[teams['season'] == 2023]
            if len(teams_2023) > 0:
                home_2023 = teams_2023[teams_2023['team'] == homeMapped]
                away_2023 = teams_2023[teams_2023['team'] == awayMapped]
                if len(home_2023) > 0:
                    home_2023 = home_2023.iloc[0]
                    if pd.isna(home_s.get('fenwickPercentage')) or home_s.get('fenwickPercentage') == 0:
                        home_s['fenwickPercentage'] = home_2023.get('fenwickPercentage', 50)
                    if pd.isna(home_s.get('shotsOnGoalFor')) or home_s.get('shotsOnGoalFor') == 0:
                        home_s['shotsOnGoalFor'] = home_2023.get('shotsOnGoalFor', 0)
                    if pd.isna(home_s.get('shotsOnGoalAgainst')) or home_s.get('shotsOnGoalAgainst') == 0:
                        home_s['shotsOnGoalAgainst'] = home_2023.get('shotsOnGoalAgainst', 0)
                    if pd.isna(home_s.get('highDangerShotsFor')) or home_s.get('highDangerShotsFor') == 0:
                        home_s['highDangerShotsFor'] = home_2023.get('highDangerShotsFor', 0)
                if len(away_2023) > 0:
                    away_2023 = away_2023.iloc[0]
                    if pd.isna(away_s.get('fenwickPercentage')) or away_s.get('fenwickPercentage') == 0:
                        away_s['fenwickPercentage'] = away_2023.get('fenwickPercentage', 50)
                    if pd.isna(away_s.get('shotsOnGoalFor')) or away_s.get('shotsOnGoalFor') == 0:
                        away_s['shotsOnGoalFor'] = away_2023.get('shotsOnGoalFor', 0)
                    if pd.isna(away_s.get('shotsOnGoalAgainst')) or away_s.get('shotsOnGoalAgainst') == 0:
                        away_s['shotsOnGoalAgainst'] = away_2023.get('shotsOnGoalAgainst', 0)
                    if pd.isna(away_s.get('highDangerShotsFor')) or away_s.get('highDangerShotsFor') == 0:
                        away_s['highDangerShotsFor'] = away_2023.get('highDangerShotsFor', 0)
        
        # Build V1 features
        features_v1 = [[
            home_s['xGoalsPercentage'], home_s['corsiPercentage'],
            away_s['xGoalsPercentage'], away_s['corsiPercentage'],
            home_s['goalsFor'], home_s['goalsAgainst'],
            away_s['goalsFor'], away_s['goalsAgainst']
        ]]
        
        # Build V2 features if needed
        if use_v2:
            features = [[
                home_s['xGoalsPercentage'], home_s['corsiPercentage'], home_s.get('fenwickPercentage', 50),
                away_s['xGoalsPercentage'], away_s['corsiPercentage'], away_s.get('fenwickPercentage', 50),
                home_s['goalsFor'], home_s['goalsAgainst'],
                away_s['goalsFor'], away_s['goalsAgainst'],
                home_s.get('shotsOnGoalFor', 0), home_s.get('shotsOnGoalAgainst', 0),
                away_s.get('shotsOnGoalFor', 0), away_s.get('shotsOnGoalAgainst', 0),
                home_s.get('highDangerShotsFor', 0), away_s.get('highDangerShotsFor', 0)
            ]]
        else:
            features = features_v1
        
        # Get V1 prediction - use appropriate features based on what model was trained on
        if model_version.startswith("v2") and not (use_both or use_both_mid or use_both_low):
            prob_v1 = float(model_v1.predict_proba(features)[0][1])
        else:
            prob_v1 = float(model_v1.predict_proba(features_v1)[0][1])
        
        # Get V2 prediction if needed
        prob_v2 = prob_v1
        if model_v2 is not None:
            prob_v2 = float(model_v2.predict_proba(features)[0][1])
        
        # For v1_v2 mode: use V1 threshold 75% and V2 threshold 70%
        if use_both or use_both_mid or use_both_low:
            pred['prob_v1'] = round(prob_v1 * 100, 1)
            pred['prob_v2'] = round(prob_v2 * 100, 1)
            pred['predicted_prob'] = round(min(prob_v1, prob_v2) * 100, 1)
            # V1_v2: V1 >= 75%, V2 >= 70%
            # V1_v2_mid: V1 >= 74%, V2 >= 68%
            # V1_v2_low: V1 >= 70%, V2 >= 65%
            if use_both_low:
                bet = prob_v1 >= 0.70 and prob_v2 >= 0.65 and pred['home_xg_pct'] >= 0.50
                mode_msg = "V1+V2 (both >= 70%/65%, xG% home >= 50%)"
            elif use_both_mid:
                bet = prob_v1 >= 0.74 and prob_v2 >= 0.68
                mode_msg = "V1+V2 (both >= 74%/68%)"
            else:
                bet = prob_v1 >= 0.75 and prob_v2 >= 0.70
                mode_msg = "V1+V2 (both >= 75%/70%)"
            
            # Check if B2B - separate for home and away
            home_b2b = pred['home_team'] in b2b_teams
            away_b2b = pred['away_team'] in b2b_teams
            pred['b2b_home'] = home_b2b
            pred['b2b_away'] = away_b2b
            
            if bet:
                if home_b2b:
                    pred['bet_recommendation'] = 'BET?'  # home tired - risky
                elif away_b2b:
                    pred['bet_recommendation'] = 'BET!'  # away tired - good
                else:
                    pred['bet_recommendation'] = 'BET'
                bet_count += 1
            else:
                pred['bet_recommendation'] = '-'
        else:
            prob = prob_v2 if model_v2 else prob_v1
            pred['predicted_prob'] = round(prob * 100, 1)
            
            # Check if B2B - separate for home and away
            home_b2b = pred['home_team'] in b2b_teams
            away_b2b = pred['away_team'] in b2b_teams
            pred['b2b_home'] = home_b2b
            pred['b2b_away'] = away_b2b
            
            if prob >= threshold_val:
                if home_b2b:
                    pred['bet_recommendation'] = 'BET?'  # home tired - risky
                elif away_b2b:
                    pred['bet_recommendation'] = 'BET!'  # away tired - good
                else:
                    pred['bet_recommendation'] = 'BET'
                bet_count += 1
            else:
                pred['bet_recommendation'] = '-'
        
        results.append(pred)
    
    mode_msg = mode_msg if (use_both or use_both_low or use_both_mid) else f"Model: {model_version}"
    return {
        "games": results[:200],
        "total_games": len(results),
        "bet_count": bet_count,
        "threshold": threshold,
        "message": f"Found {len(results)} games, {bet_count} qualify for betting. {mode_msg}"
    }


def save_model(model, version: str):
    path = MODELS_PATH / f"model_{version}.pkl"
    joblib.dump(model, path)
    return str(path)


def load_model(version: str):
    path = MODELS_PATH / f"model_{version}.pkl"
    if path.exists():
        return joblib.load(path)
    return None


@app.get("/models")
async def list_models():
    available = []
    for f in MODELS_PATH.glob("model_*.pkl"):
        available.append(f.stem.replace("model_", ""))
    # Always return v1, v2, v1_v2, v1_v2_mid and v1_v2_low as options
    available = ["v1", "v2", "v1_v2", "v1_v2_mid", "v1_v2_low"]
    return {"models": available}


@app.post("/models/{version}/train")
async def train_and_save_model(version: str):
    global combined_df
    if combined_df is None:
        all_data = [df]
        if df_2024_25 is not None:
            all_data.append(df_2024_25)
        if df_2025_26 is not None:
            all_data.append(df_2025_26)
        combined_df = pd.concat(all_data, ignore_index=True)
    
    train_seasons = [20132014, 20142015, 20152016, 20162017, 20172018, 20182019, 20192020, 20202021, 20212022, 20222023, 20232024, 20242025]
    train_df = combined_df[combined_df['season'].isin(train_seasons)].copy()
    train_df['home_3_plus'] = (train_df['home_gf'] >= 3).astype(int)
    
    features = feature_cols_v2 if version.startswith("v2") else feature_cols
    
    X_train = train_df[features]
    y_train = train_df['home_3_plus']
    
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    path = save_model(model, version)
    return {"status": "saved", "version": version, "path": path, "train_samples": len(X_train)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
