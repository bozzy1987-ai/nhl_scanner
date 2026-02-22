#!/usr/bin/env python3
"""
Fetch real NHL game data from the NHL API and calculate advanced stats
"""
import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os

TEAM_ID_TO_ABBREV = {
    1: 'NJD', 2: 'NYI', 3: 'NYR', 4: 'PHI', 5: 'PIT', 6: 'BOS', 7: 'BUF',
    8: 'MTL', 9: 'OTT', 10: 'TOR', 11: 'DET', 12: 'FLA', 13: 'TBL', 14: 'CAR',
    15: 'WSH', 16: 'CHI', 17: 'VGK', 18: 'ANA', 19: 'DAL', 20: 'EDM', 21: 'CGY',
    22: 'LAK', 23: 'SJS', 24: 'VAN', 25: 'WPG', 26: 'ARI', 27: 'PHX', 28: 'MIN',
    29: 'STL', 30: 'CBJ', 31: 'CGY', 32: 'SEA', 33: 'UTA', 34: 'NSH', 52: 'CLE',
    53: 'JER', 54: 'SCB', 55: 'LDN', 56: 'MAN', 57: 'NSC', 58: 'ADG', 59: 'ANC'
}

ABBREV_TO_TEAM_ID = {v: k for k, v in TEAM_ID_TO_ABBREV.items()}

def get_schedule(date_str):
    """Get NHL schedule for a specific date"""
    url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error fetching schedule for {date_str}: {e}")
    return None

def get_play_by_play(game_id):
    """Get play-by-play data for a game"""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Error fetching play-by-play for {game_id}: {e}")
    return None

def calculate_game_stats(pbp_data, home_score=0, away_score=0):
    """Calculate CF%, xGF% and other stats from play-by-play"""
    if not pbp_data:
        return None
    
    home_abbrev = pbp_data.get('homeTeam', {}).get('abbrev', '')
    away_abbrev = pbp_data.get('awayTeam', {}).get('abbrev', '')
    
    if not home_abbrev or not away_abbrev:
        return None
    
    home_id = pbp_data.get('homeTeam', {}).get('id', 0)
    away_id = pbp_data.get('awayTeam', {}).get('id', 0)
    
    home_cf = 0
    away_cf = 0
    home_xgf = 0
    away_xgf = 0
    home_hdcf = 0
    away_hdcf = 0
    home_sf = 0
    away_sf = 0
    
    for play in pbp_data.get('plays', []):
        play_type = play.get('typeDescKey', '')
        if play_type not in ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']:
            continue
        
        team_id = play.get('details', {}).get('eventOwnerTeamId', 0)
        
        if team_id == home_id:
            home_cf += 1
            home_xgf += 1
            if play_type in ['shot-on-goal', 'goal']:
                home_sf += 1
                x = play.get('details', {}).get('xCoord', 0)
                if -80 < x < 80:
                    home_hdcf += 1
        elif team_id == away_id:
            away_cf += 1
            away_xgf += 1
            if play_type in ['shot-on-goal', 'goal']:
                away_sf += 1
                x = play.get('details', {}).get('xCoord', 0)
                if -80 < x < 80:
                    away_hdcf += 1
    
    total_cf = home_cf + away_cf
    total_xgf = home_xgf + away_xgf
    
    if total_cf > 0:
        home_cf_pct = (home_cf / total_cf) * 100
        away_cf_pct = (away_cf / total_cf) * 100
    else:
        home_cf_pct = 50
        away_cf_pct = 50
    
    if total_xgf > 0:
        home_xgf_pct = (home_xgf / total_xgf) * 100
        away_xgf_pct = (away_xgf / total_xgf) * 100
    else:
        home_xgf_pct = 50
        away_xgf_pct = 50
    
    total_sf = home_sf + away_sf
    if total_sf > 0:
        home_sf_pct = (home_sf / total_sf) * 100
        away_sf_pct = (away_sf / total_sf) * 100
    else:
        home_sf_pct = 50
        away_sf_pct = 50
    
    home_gf = home_score
    away_gf = away_score
    
    total_gf = home_gf + away_gf
    if total_gf > 0:
        home_gf_pct = (home_gf / total_gf) * 100
        away_gf_pct = (away_gf / total_gf) * 100
    else:
        home_gf_pct = 50
        away_gf_pct = 50
    
    return {
        'home_cf_pct': home_cf_pct,
        'away_cf_pct': away_cf_pct,
        'home_xgf_pct': home_xgf_pct,
        'away_xgf_pct': away_xgf_pct,
        'home_hdcf_pct': home_hdcf / max(1, home_hdcf + away_hdcf) * 100 if (home_hdcf + away_hdcf) > 0 else 50,
        'away_hdcf_pct': away_hdcf / max(1, home_hdcf + away_hdcf) * 100 if (home_hdcf + away_hdcf) > 0 else 50,
        'home_sf_pct': home_sf_pct,
        'away_sf_pct': away_sf_pct,
        'home_gf_pct': home_gf_pct,
        'away_gf_pct': away_gf_pct,
        'home_gf': home_gf,
        'away_gf': away_gf,
        'home_team': home_abbrev,
        'away_team': away_abbrev,
    }

def fetch_season_games(season_start_year, max_games_per_day=5):
    """Fetch all games for a season"""
    season = f"{season_start_year}{season_start_year + 1}"
    
    start_date = datetime(season_start_year, 10, 1)
    if season_start_year >= 2024:
        start_date = datetime(season_start_year, 10, 1)
    
    end_date = datetime(season_start_year + 1, 4, 20)
    
    games = []
    current = start_date
    game_ids = set()
    
    print(f"Fetching season {season} from {start_date.date()} to {end_date.date()}")
    
    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        schedule = get_schedule(date_str)
        
        if schedule and schedule.get('gameWeek'):
            for day in schedule['gameWeek']:
                for game in day.get('games', []):
                    game_id = game.get('id')
                    if game_id and game_id not in game_ids:
                        game_ids.add(game_id)
                        
                        season_id = game.get('season')
                        if season_id:
                            season_code = str(season_id)
                        else:
                            season_code = season
                        
                        print(f"  Fetching game {game_id}: {game.get('homeTeam',{}).get('abbrev','?')} vs {game.get('awayTeam',{}).get('abbrev','?')}")
                        
                        time.sleep(0.2)
                        
                        pbp = get_play_by_play(game_id)
                        stats = calculate_game_stats(pbp)
                        
                        if stats:
                            stats['season'] = season_code
                            stats['date'] = date_str
                            stats['game_id'] = game_id
                            games.append(stats)
        
        current += timedelta(days=1)
        
        if len(games) % 50 == 0 and len(games) > 0:
            print(f"  Fetched {len(games)} games so far...")
    
    return games

def main():
    os.makedirs('data', exist_ok=True)
    
    # Sample: 2 teams, 2 seasons
    sample_teams = ['BOS', 'MTL']  # Boston, Montreal
    seasons = ['20232024', '20242025']
    
    all_games = []
    fetched_game_ids = set()  # Track already fetched games
    
    for season in seasons:
        year = int(season[:4])
        
        if season == '20232024':
            start_date = datetime(2023, 10, 1)
            end_date = datetime(2024, 4, 15)
        else:
            start_date = datetime(2024, 10, 1)
            end_date = datetime(2025, 4, 15)
        
        print(f"\nFetching season {season} ({start_date.date()} to {end_date.date()})")
        current = start_date
        
        while current <= end_date:
            date_str = current.strftime('%Y-%m-%d')
            schedule = get_schedule(date_str)
            
            if schedule and schedule.get('gameWeek'):
                for day in schedule['gameWeek']:
                    for game in day.get('games', []):
                        home = game.get('homeTeam', {}).get('abbrev', '')
                        away = game.get('awayTeam', {}).get('abbrev', '')
                        
                        if home in sample_teams or away in sample_teams:
                            game_id = game.get('id')
                            if game_id and game_id not in fetched_game_ids:
                                fetched_game_ids.add(game_id)
                                home_score = game.get('homeTeam', {}).get('score', 0)
                                away_score = game.get('awayTeam', {}).get('score', 0)
                                print(f"  Game {game_id}: {home} vs {away} ({home_score}-{away_score})")
                                
                                pbp = get_play_by_play(game_id)
                                stats = calculate_game_stats(pbp, home_score, away_score)
                                
                                if stats:
                                    stats['season'] = season
                                    stats['date'] = date_str
                                    stats['game_id'] = game_id
                                    all_games.append(stats)
            
            time.sleep(0.2)
            current += timedelta(days=1)
    
    if all_games:
        df = pd.DataFrame(all_games)
        df['home_goals_2_5'] = (df['home_gf'] >= 3).astype(int)
        df.to_csv('data/nhl_games.csv', index=False)
        print(f"\nTotal: {len(df)} games saved")
        print(f"Teams: {df['home_team'].unique()}")
        print(f"Target: {df['home_goals_2_5'].mean():.2%} with 3+ goals")
    else:
        print("No games fetched!")

if __name__ == '__main__':
    main()
