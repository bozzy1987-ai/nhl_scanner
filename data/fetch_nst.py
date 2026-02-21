#!/usr/bin/env python3
"""
Script to fetch NHL game data from Natural Stat Trick
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime

TEAMS = [
    'ANA', 'BOS', 'BUF', 'CAR', 'CBJ', 'CGY', 'CHI', 'COL', 'DAL', 'DET',
    'EDM', 'FLA', 'L.A', 'MIN', 'MTL', 'N.J', 'NSH', 'NYI', 'NYR', 'OTT',
    'PHI', 'PIT', 'S.J', 'SEA', 'STL', 'T.B', 'TOR', 'UTA', 'VAN', 'VGK', 'WPG', 'WSH'
]

SEASONS = [
    '20132014', '20142015', '20152016', '20162017', '20172018', '20182019',
    '20192020', '20202021', '20212022', '20222023', '20232024'
]

def fetch_team_games(team: str, season: str) -> list:
    """Fetch games for a specific team and season"""
    url = f"https://www.naturalstattrick.com/games.php?team={team}&fromseason={season}&thruseason={season}&stype=2&sit=5v5"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return parse_games_html(response.text, team, season)
    except Exception as e:
        print(f"Error fetching {team} {season}: {e}")
        return []

def parse_games_html(html: str, team: str, season: str) -> list:
    """Parse HTML to extract game data"""
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', {'id': 'teams'})
    
    if not table:
        return []
    
    games = []
    rows = table.find('tbody').find_all('tr')
    
    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 2:
            continue
        
        game_cell = cols[0].get_text(strip=True)
        match = re.search(r'(\d{4}-\d{2}-\d{2})', game_cell)
        if not match:
            continue
        
        date = match.group(1)
        result_match = re.search(r'(\w+)\s+(\d+),\s+(\w+)\s+(\d+)', game_cell)
        
        try:
            gf = int(cols[10].get_text(strip=True)) if cols[10].get_text(strip=True) else 0
            ga = int(cols[11].get_text(strip=True)) if cols[11].get_text(strip=True) else 0
            
            cf = int(cols[3].get_text(strip=True)) if cols[3].get_text(strip=True) else 0
            ca = int(cols[4].get_text(strip=True)) if cols[4].get_text(strip=True) else 0
            cf_pct = float(cols[5].get_text(strip=True).replace('%', '')) if cols[5].get_text(strip=True) else 0
            
            ff = int(cols[6].get_text(strip=True)) if cols[6].get_text(strip=True) else 0
            fa = int(cols[7].get_text(strip=True)) if cols[7].get_text(strip=True) else 0
            ff_pct = float(cols[8].get_text(strip=True).replace('%', '')) if cols[8].get_text(strip=True) else 0
            
            sf = int(cols[9].get_text(strip=True)) if cols[9].get_text(strip=True) else 0
            sa = int(cols[10].get_text(strip=True)) if len(cols) > 10 and cols[10].get_text(strip=True) else 0
            
            xgf = float(cols[12].get_text(strip=True)) if cols[12].get_text(strip=True) else 0
            xga = float(cols[13].get_text(strip=True)) if cols[13].get_text(strip=True) else 0
            xgf_pct = float(cols[14].get_text(strip=True).replace('%', '')) if cols[14].get_text(strip=True) else 0
            
            hdtf = int(cols[15].get_text(strip=True)) if cols[15].get_text(strip=True) else 0
            hdca = int(cols[16].get_text(strip=True)) if cols[16].get_text(strip=True) else 0
            hdtf_pct = float(cols[17].get_text(strip=True).replace('%', '')) if cols[17].get_text(strip=True) else 0
            
            games.append({
                'season': season,
                'date': date,
                'team': team,
                'gf': gf,
                'ga': ga,
                'cf': cf,
                'ca': ca,
                'cf_pct': cf_pct,
                'ff': ff,
                'fa': fa,
                'ff_pct': ff_pct,
                'sf': sf,
                'sa': sa,
                'xgf': xgf,
                'xga': xga,
                'xgf_pct': xgf_pct,
                'hdtf': hdtf,
                'hdca': hdca,
                'hdtf_pct': hdtf_pct
            })
        except (IndexError, ValueError) as e:
            continue
    
    return games

def main():
    all_games = []
    
    for season in SEASONS:
        print(f"\n=== Fetching season {season} ===")
        for i, team in enumerate(TEAMS):
            print(f"  [{i+1}/{len(TEAMS)}] {team}", end='\r')
            games = fetch_team_games(team, season)
            all_games.extend(games)
            time.sleep(0.5)
        
        df = pd.DataFrame(all_games)
        df.to_csv(f'nhl_games_{season}.csv', index=False)
        print(f"\nSeason {season}: {len(games)} games fetched")
    
    df = pd.DataFrame(all_games)
    df.to_csv('nhl_all_games.csv', index=False)
    print(f"\nTotal: {len(df)} games saved to nhl_all_games.csv")

if __name__ == '__main__':
    main()
