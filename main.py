#!/usr/bin/env python3
"""
FastAPI backend for NHL Hockey Analytics
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = FastAPI(title="Hockey Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "model" / "hockey_model.pkl"
DATA_PATH = Path(__file__).parent / "data" / "nhl_games.csv"

model = None
df = None

@app.on_event("startup")
async def load_model():
    global model, df
    try:
        model = joblib.load(MODEL_PATH)
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")

class SimulationRequest(BaseModel):
    train_season_start: int
    train_season_end: int
    test_season_start: int
    test_season_end: int
    confidence_threshold: float
    bet_amount: float

class SimulationResult(BaseModel):
    total_profit: float
    roi_percent: float
    hit_rate: float
    total_matches: int
    matched_bets: int
    bankroll_history: list
    bets: list

def prepare_features_for_prediction(home_cf, away_cf, home_xgf, away_xgf, home_hdcf, away_hdcf, home_sf, away_sf, home_gf, away_gf):
    """Prepare features for model prediction"""
    features = pd.DataFrame({
        'home_cf_pct': [home_cf],
        'away_cf_pct': [away_cf],
        'cf_diff': [home_cf - away_cf],
        'home_xgf_pct': [home_xgf],
        'away_xgf_pct': [away_xgf],
        'xgf_diff': [home_xgf - away_xgf],
        'home_hdcf_pct': [home_hdcf],
        'away_hdcf_pct': [away_hdcf],
        'hdcf_diff': [home_hdcf - away_hdcf],
        'home_sf_pct': [home_sf],
        'away_sf_pct': [away_sf],
        'sf_diff': [home_sf - away_sf],
        'home_gf_pct': [home_gf],
        'away_gf_pct': [away_gf],
        'gf_diff': [home_gf - away_gf],
        'home_advantage': [1]
    })
    return features

@app.get("/")
async def root():
    return {"message": "Hockey Analytics API", "version": "1.0.0"}

@app.get("/seasons")
async def get_seasons():
    """Get available seasons"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    seasons = sorted(df['season'].unique())
    return {"seasons": [s for s in seasons if int(s[:4]) <= 2024]}

@app.post("/simulate")
async def simulate(request: SimulationRequest):
    """Run backtesting simulation"""
    if model is None or df is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    train_years = list(range(request.train_season_start, request.train_season_end + 1))
    test_years = list(range(request.test_season_start, request.test_season_end + 1))
    
    train_seasons = [f"{y}{y+1}" for y in train_years if y < 2024]
    test_seasons = [f"{y}{y+1}" for y in test_years if y < 2024]
    
    test_df = df[df['season'].isin(test_seasons)].copy()
    
    if len(test_df) == 0:
        raise HTTPException(status_code=400, detail="No test data available for selected seasons")
    
    X_test = prepare_features_for_prediction(
        test_df['home_cf_pct'].values,
        test_df['away_cf_pct'].values,
        test_df['home_xgf_pct'].values,
        test_df['away_xgf_pct'].values,
        test_df['home_hdcf_pct'].values,
        test_df['away_hdcf_pct'].values,
        test_df['home_sf_pct'].values,
        test_df['away_sf_pct'].values,
        test_df['home_gf_pct'].values,
        test_df['away_gf_pct'].values
    )
    
    probabilities = model.predict_proba(X_test)[:, 1]
    test_df['predicted_prob'] = probabilities
    test_df['actual_1_5'] = test_df['home_goals_1_5']
    
    qualifying_bets = test_df[test_df['predicted_prob'] >= request.confidence_threshold / 100].copy()
    
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
    
    qualifying_bets['won'] = (qualifying_bets['actual_1_5'] == 1).astype(int)
    qualifying_bets['profit'] = np.where(
        qualifying_bets['won'] == 1,
        request.bet_amount * 0.9,
        -request.bet_amount
    )
    
    bankroll = 0.0
    bankroll_history = []
    for idx, row in qualifying_bets.iterrows():
        bankroll += row['profit']
        bankroll_history.append({
            "match": f"{row['home_team']} vs {row['away_team']}",
            "date": row['date'],
            "predicted_prob": round(row['predicted_prob'] * 100, 1),
            "result": "WIN" if row['won'] == 1 else "LOSS",
            "profit": round(row['profit'], 2),
            "bankroll": round(bankroll, 2)
        })
    
    total_profit = bankroll
    total_staked = len(qualifying_bets) * request.bet_amount
    roi_percent = (total_profit / total_staked * 100) if total_staked > 0 else 0
    hit_rate = qualifying_bets['won'].mean() * 100 if len(qualifying_bets) > 0 else 0
    
    comment = generate_comment(roi_percent, hit_rate, len(qualifying_bets), request.confidence_threshold, qualifying_bets)
    
    return {
        "total_profit": round(total_profit, 2),
        "roi_percent": round(roi_percent, 2),
        "hit_rate": round(hit_rate, 2),
        "total_matches": len(test_df),
        "matched_bets": len(qualifying_bets),
        "bankroll_history": bankroll_history,
        "bets": [
            {
                "match": b["match"],
                "date": b["date"],
                "predicted_prob": b["predicted_prob"],
                "result": b["result"],
                "profit": b["profit"]
            }
            for b in bankroll_history
        ],
        "comment": comment
    }

def generate_comment(roi, hit_rate, num_bets, threshold, bets_df):
    """Generate Claude-style comment in Polish"""
    if num_bets == 0:
        return "Brak meczów spełniających wybrany próg confidence. Spróbuj obniżyć próg."
    
    comment = f"Model osiągnął ROI {roi:.1f}% typując mecze z confidence powyżej {threshold}%. "
    
    if hit_rate >= 60:
        comment += "Bardzo dobry hit rate! "
    elif hit_rate >= 50:
        comment += "Przyzwoity hit rate. "
    else:
        comment += "Niski hit rate - model może wymagać dostrojenia. "
    
    if len(bets_df) > 0:
        high_cf_bets = bets_df[bets_df['home_cf_pct'] > 55]
        if len(high_cf_bets) > 0:
            high_cf_hit_rate = high_cf_bets['won'].mean() * 100
            comment += f"Najlepiej radził sobie w meczach gdzie gospodarz miał CF% > 55% (hit rate: {high_cf_hit_rate:.1f}%). "
    
    if roi > 0:
        comment += "Strategia przyniosła zysk!"
    else:
        comment += "Strategia przyniosła stratę - rozważ zmianę parametrów."
    
    return comment

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
