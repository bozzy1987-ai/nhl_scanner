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
from pathlib import Path
import json

DATA_PATH = Path(__file__).parent / "data" / "nhl_game_results.csv"
TEAM_STATS_PATH = Path(__file__).parent.parent / "teams_2008_to_2024.csv"

feature_cols = ['home_xg_pct', 'home_cf_pct', 'away_xg_pct', 'away_cf_pct',
                'home_goals_for', 'home_goals_against', 'away_goals_for', 'away_goals_against']

df = None
teams = None

def load_data():
    global df, teams
    try:
        # Load game results
        games = pd.read_csv(DATA_PATH)
        games['season'] = games['season'].astype(int)
        
        # Load team stats
        teams = pd.read_csv(TEAM_STATS_PATH)
        teams = teams[teams['situation'] == 'all']
        
        # Fix team names
        games['home_team'] = games['home_team'].replace('PHX', 'ARI')
        games['away_team'] = games['away_team'].replace('PHX', 'ARI')
        
        # Build features
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
            
            features.append({
                'season': row['season'],
                'date': row['date'],
                'home_team': home,
                'away_team': away,
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
        
        df = pd.DataFrame(features)
        print(f"Loaded {len(df)} games with features")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
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

@app.get("/")
async def root():
    return {"message": "Hockey Analytics API v3.0 - Dynamic Training"}

@app.get("/seasons")
async def get_seasons():
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    seasons = sorted(df['season'].unique())
    return {"seasons": [int(s) for s in seasons]}

@app.post("/simulate")
async def simulate(request: SimulationRequest):
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Convert years to season codes
    def year_to_season_code(year):
        return year * 10000 + (year + 1) % 10000
    
    train_seasons = [year_to_season_code(y) for y in range(request.train_season_start, request.train_season_end + 1)]
    test_seasons = [year_to_season_code(y) for y in range(request.test_season_start, request.test_season_end + 1)]
    
    # Filter data
    train_df = df[df['season'].isin(train_seasons)]
    test_df = df[df['season'].isin(test_seasons)].copy()
    
    if len(train_df) == 0:
        raise HTTPException(status_code=400, detail="Brak danych treningowych")
    if len(test_df) == 0:
        raise HTTPException(status_code=400, detail="Brak danych testowych")
    
    # Create target
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['home_3_plus'] = (train_df['home_gf'] >= 3).astype(int)
    test_df['home_3_plus'] = (test_df['home_gf'] >= 3).astype(int)
    
    # Train model dynamically
    X_train = train_df[feature_cols]
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
    X_test = test_df[feature_cols]
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
        request.bet_amount * 0.6,
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
