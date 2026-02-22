#!/usr/bin/env python3
"""
FastAPI backend for NHL Hockey Analytics - z prawdziwymi danymi
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model" / "hockey_model.pkl"
DATA_PATH = Path(__file__).parent / "data" / "nhl_game_results.csv"

model = None
df = None
feature_cols = ['home_xg_pct', 'home_cf_pct', 'away_xg_pct', 'away_cf_pct',
                'home_goals_for', 'home_goals_against', 'away_goals_for', 'away_goals_against']

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, df
    try:
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
        df['season'] = df['season'].astype(str)
        print(f"Loaded model and {len(df)} games")
    except Exception as e:
        print(f"Error loading: {e}")
    yield

app = FastAPI(title="Hockey Analytics API", version="2.0", lifespan=lifespan)

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
    return {"message": "Hockey Analytics API v2.0", "version": "2.0.0"}

@app.get("/seasons")
async def get_seasons():
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    seasons = sorted(df['season'].unique())
    return {"seasons": [int(str(s)[:4]) for s in seasons]}

@app.post("/simulate")
async def simulate(request: SimulationRequest):
    if model is None or df is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Get test seasons
    test_years = list(range(request.test_season_start, request.test_season_end + 1))
    test_seasons = [str(y) + str(y+1) for y in test_years]
    
    test_df = df[df['season'].isin(test_seasons)].copy()
    
    if len(test_df) == 0:
        raise HTTPException(status_code=400, detail="Brak danych dla wybranych sezonów")
    
    # Make predictions
    X_test = test_df[feature_cols].copy()
    probabilities = model.predict_proba(X_test)[:, 1]
    test_df['predicted_prob'] = probabilities
    
    # Target: home team scores 3+ goals
    test_df['actual_3_plus'] = (test_df['home_gf'] >= 3).astype(int)
    
    # Filter by confidence threshold
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
            "comment": "Brak meczów spełniających próg confidence. Spróbuj obniżyć próg."
        }
    
    # Calculate profit (60 PLN win for 100 PLN bet)
    qualifying_bets['won'] = (qualifying_bets['actual_3_plus'] == 1).astype(int)
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
        "comment": f"ROI: {roi_percent:.1f}%, Hit rate: {hit_rate:.1f}% przy progu {request.confidence_threshold}%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
