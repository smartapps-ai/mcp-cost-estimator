import React, { useState } from 'react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL ?? '';

const GPT_MODELS = [
  { value: 'gpt-5.4', label: 'gpt-5.4' },
];

// Dataset → available servers (mirrors backend DATASET_SERVERS)
const DATASET_SERVERS = {
  unitus: ['sql_server', 'tursio', 'supabase'],
  umcu:   ['snowflake',  'tursio', 'supabase'],
  tpch:   ['tursio', 'supabase'],
};

const DATASET_LABELS = {
  unitus: 'Unitus',
  umcu:   'UMCU',
  tpch:   'TPC-H',
};

const FEATURE_LABELS = {
  domain:      'Domain',
  category:    'Category',
  complexity:  'Complexity',
  result_size: 'Result Size',
  intent:      'Intent',
  answer_type: 'Answer Type',
};

function PlatformCard({ platform, data }) {
  return (
    <div className="platform-card">
      <h3>{platform.replace(/_/g, ' ').toUpperCase()}</h3>
      <div className="token-row">
        <span className="token-label">Input</span>
        <span className="token-value">{data.input_tokens.toLocaleString()}</span>
      </div>
      <div className="token-row">
        <span className="token-label">Output</span>
        <span className="token-value">{data.output_tokens.toLocaleString()}</span>
      </div>
      <div className="token-row token-total">
        <span className="token-label">Total</span>
        <span className="token-value">{data.total_tokens.toLocaleString()}</span>
      </div>
      <div className="cost">${data.cost_usd.toFixed(6)}</div>
    </div>
  );
}

function App() {
  const [question, setQuestion] = useState('');
  const [gptModel, setGptModel] = useState('gpt-5.4');
  const [dataset, setDataset] = useState('unitus');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) {
      setError('Please enter a question.');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch(`${API_BASE}/estimate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed, gpt_model: gptModel, dataset }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Something went wrong on the server.');
      }

      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>MCP Server Cost Estimator</h1>
        <p>
          Enter a question to route through the Model Context Protocol (MCP). The backend
          infers intent, runs per-platform regression models trained on real agent
          response data, and estimates input/output token usage and cost.
        </p>
      </header>

      <form onSubmit={handleSubmit} className="estimator-form">
        <div className="form-group">
          <label htmlFor="question">User Question</label>
          <textarea
            id="question"
            rows="4"
            placeholder="e.g., What percentage of dormant accounts have inactive loans or cards?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label htmlFor="dataset">Dataset</label>
          <select
            id="dataset"
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
          >
            {Object.entries(DATASET_LABELS).map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="gptModel">Agent Model</label>
          <select
            id="gptModel"
            value={gptModel}
            onChange={(e) => setGptModel(e.target.value)}
          >
            {GPT_MODELS.map(({ value, label }) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Estimating…' : 'Estimate Cost'}
        </button>
      </form>

      {error && <div className="error-message">Error: {error}</div>}

      {results && (
        <div className="results-container">
          <section className="results-section">
            <h2>Inferred Features</h2>
            <div className="metrics-grid">
              {Object.entries(results.inferred_features).map(([key, value]) => (
                <div className="metric-card" key={key}>
                  <h3>{FEATURE_LABELS[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</h3>
                  <p>{String(value).replace(/_/g, ' ')}</p>
                </div>
              ))}
            </div>
          </section>

          <hr className="divider" />

          <section className="results-section">
            <h2>Cost Estimates by Platform</h2>
            <div className="platform-grid">
              {Object.entries(results.estimates).map(([platform, data]) => (
                <PlatformCard key={platform} platform={platform} data={data} />
              ))}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

export default App;
