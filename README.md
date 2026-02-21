# NHL Hockey Analytics - Symulator Backtestingu

Aplikacja do symulacji i testowania strategii typowania wyników meczów NHL z wykorzystaniem modelu ML XGBoost.

## Funkcjonalności

- Pobieranie danych z Natural Stat Trick
- Trening modelu XGBoost do przewidywania wyników meczów
- Symulator backtestingu z konfigurowalnymi parametrami
- Wykresy historia bankrolla
- Interfejs w języku polskim

## Stack technologiczny

- **Backend**: FastAPI (Python)
- **Frontend**: React + Recharts
- **ML**: XGBoost
- **Konteneryzacja**: Docker
- **Orchestracja**: Kubernetes (k3s)
- **CI/CD**: GitHub Actions

## Struktura projektu

```
nhl_scanner/
├── backend/
│   ├── main.py          # FastAPI application
│   ├── Dockerfile       # Backend Docker image
│   ├── model/
│   │   ├── train.py     # Model training script
│   │   └── hockey_model.pkl
│   └── data/
│       ├── nhl_games.csv
│       └── fetch_nst.py
├── frontend/
│   ├── src/
│   │   ├── App.js       # Main React component
│   │   ├── App.css      # Styles
│   │   └── index.js
│   ├── public/
│   │   └── index.html
│   ├── package.json
│   └── Dockerfile
├── helm/
│   └── hockey-analytics/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
└── .github/
    └── workflows/
        └── ci.yml
```

## Uruchomienie lokalne

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend

```bash
cd frontend
npm install
npm start
```

## API

### POST /simulate

Uruchomienie symulacji backtestingu.

**Request:**
```json
{
  "train_season_start": 2013,
  "train_season_end": 2020,
  "test_season_start": 2021,
  "test_season_end": 2023,
  "confidence_threshold": 65,
  "bet_amount": 100
}
```

**Response:**
```json
{
  "total_profit": 150.0,
  "roi_percent": 12.5,
  "hit_rate": 58.3,
  "total_matches": 120,
  "matched_bets": 24,
  "bankroll_history": [...],
  "bets": [...],
  "comment": "Model osiągnął ROI..."
}
```

## Deployment na k3s

```bash
helm install hockey-analytics ./helm/hockey-analytics -n nhl-scanner
```

## Konfiguracja CI/CD

Wymagane sekrety w GitHub:
- `HARBOR_USER` - użytkownik Harbor
- `HARBOR_PASSWORD` - hasło Harbor
- `SSH_KEY` - klucz SSH do k3s
