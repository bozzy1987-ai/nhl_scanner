import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [seasons, setSeasons] = useState([]);
  const [trainStart, setTrainStart] = useState(2013);
  const [trainEnd, setTrainEnd] = useState(2020);
  const [testStart, setTestStart] = useState(2021);
  const [testEnd, setTestEnd] = useState(2023);
  const [confidence, setConfidence] = useState(65);
  const [betAmount, setBetAmount] = useState(100);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setSeasons([2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]);
  }, []);

  const runSimulation = async () => {
    setLoading(true);
    setError(null);
    setResults(null);
    
    try {
      const response = await axios.post(`${API_URL}/simulate`, {
        train_season_start: trainStart,
        train_season_end: trainEnd,
        test_season_start: testStart,
        test_season_end: testEnd,
        confidence_threshold: confidence,
        bet_amount: betAmount
      });
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Wystąpił błąd podczas symulacji');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Hockey Analytics - Symulator Backtestingu</h1>
        <p>Model ML do typowania wyników NHL</p>
      </header>

      <main className="App-main">
        <div className="controls">
          <h2>Konfiguracja Symulacji</h2>
          
          <div className="control-group">
            <label>Sezon treningowy (początek):</label>
            <select value={trainStart} onChange={(e) => setTrainStart(Number(e.target.value))}>
              {seasons.map(s => <option key={s} value={s}>{s}/{s+1}</option>)}
            </select>
          </div>
          
          <div className="control-group">
            <label>Sezon treningowy (koniec):</label>
            <select value={trainEnd} onChange={(e) => setTrainEnd(Number(e.target.value))}>
              {seasons.map(s => <option key={s} value={s}>{s}/{s+1}</option>)}
            </select>
          </div>
          
          <div className="control-group">
            <label>Sezon testowy (początek):</label>
            <select value={testStart} onChange={(e) => setTestStart(Number(e.target.value))}>
              {seasons.map(s => <option key={s} value={s}>{s}/{s+1}</option>)}
            </select>
          </div>
          
          <div className="control-group">
            <label>Sezon testowy (koniec):</label>
            <select value={testEnd} onChange={(e) => setTestEnd(Number(e.target.value))}>
              {seasons.map(s => <option key={s} value={s}>{s}/{s+1}</option>)}
            </select>
          </div>
          
          <div className="control-group">
            <label>Próg confidence (%): {confidence}%</label>
            <input 
              type="range" 
              min="50" 
              max="95" 
              value={confidence} 
              onChange={(e) => setConfidence(Number(e.target.value))}
            />
          </div>
          
          <div className="control-group">
            <label>Kwota na typ (zł):</label>
            <input 
              type="number" 
              value={betAmount} 
              onChange={(e) => setBetAmount(Number(e.target.value))}
              min="1"
            />
          </div>
          
          <button 
            className="run-button" 
            onClick={runSimulation}
            disabled={loading}
          >
            {loading ? 'Uruchamianie symulacji...' : 'Uruchom Symulację'}
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        {results && (
          <div className="results">
            <h2>Wyniki Symulacji</h2>
            
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-value">{results.total_profit} zł</div>
                <div className="stat-label">Zysk/Strata</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{results.roi_percent}%</div>
                <div className="stat-label">ROI</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{results.hit_rate}%</div>
                <div className="stat-label">Hit Rate</div>
              </div>
              <div className="stat-card">
                <div className="stat-value">{results.matched_bets}/{results.total_matches}</div>
                <div className="stat-label">Typy spełniające próg</div>
              </div>
            </div>
            
            <div className="comment">
              <h3>Komentarz Modelu</h3>
              <p>{results.comment}</p>
            </div>
            
            {results.bankroll_history && results.bankroll_history.length > 0 && (
              <div className="chart">
                <h3>Historia Bankrolla</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={results.bankroll_history}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{fontSize: 12}} />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="bankroll" stroke="#8884d8" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            
            {results.bets && results.bets.length > 0 && (
              <div className="bets-table">
                <h3>Szczegóły Typów</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Data</th>
                      <th>Mecz</th>
                      <th>Przewidywane %</th>
                      <th>Wynik</th>
                      <th>Zysk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.bets.map((bet, idx) => (
                      <tr key={idx} className={bet.result === 'WIN' ? 'win' : 'loss'}>
                        <td>{bet.date}</td>
                        <td>{bet.match}</td>
                        <td>{bet.predicted_prob}%</td>
                        <td>{bet.result}</td>
                        <td className={bet.profit >= 0 ? 'profit' : 'loss'}>{bet.profit} zł</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
