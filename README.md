# MCP Cost Estimator

Estimates token usage and API cost for natural-language questions routed through a
[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server.

The backend classifies each question by **domain**, **complexity**, **intent**, **category**, **result_size**, and **answer_type** (using
an OpenAI LLM or keyword fallback), then runs pre-trained regression models to
predict token consumption across four database platforms: **SQL Server**, **Tursio**, and
**Snowflake**, **Supabase**.

---

## Architecture

```
mcp-cost-estimator/
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── inference.py     # Feature extraction + ML inference
│   └── models/          # Serialised scikit-learn models (.joblib)
├── frontend/            # React + Vite UI
├── train_model.py       # Script to (re-)create the regression models
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── README.md
```

---

## Setup

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| Node.js | 18+ |

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

The `OPENAI_API_KEY` is optional. If it is not set, the backend falls back to
keyword-based question classification.

### 3. Train the models

Run once before starting the server (or whenever you want to retrain):

```bash
python train_model.py
```

This writes `backend/models/{sql_server,tursio,snowflake,supabase}_model.joblib`.

### 4. Start the backend

```bash
uvicorn backend.main:app --reload
```

The API is available at `http://127.0.0.1:8000`.
Check `http://127.0.0.1:8000/health` to verify models are loaded.

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

The UI is available at `http://localhost:5173`. The dev server proxies
`/estimate` and `/health` requests to the backend automatically.

---

## API Reference

### `POST /estimate`

Request body:

```json
{
  "question": "What is the total revenue by product category this quarter?",
  "gpt_model": "gpt-5.4"
}
```

Valid `gpt_model` values: `gpt-5.4`.

Response:

```json
{
  "inferred_features": {
    "domain": "banking",
    "complexity": "medium",
    "intent": "analyze"
  },
  "estimates": {
    "sql_server": { "tokens": 720, "cost_usd": 0.00720 },
    "tursio":     { "tokens": 648, "cost_usd": 0.00648 },
    "snowflake":  { "tokens": 1160, "cost_usd": 0.01160 }
  }
}
```

### `GET /health`

Returns loaded model names and server status.

---

## Production Deployment

### Backend

Set `OPENAI_API_KEY` in your environment and start with a
production ASGI server:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Update `allow_origins` in `backend/main.py` to your frontend's exact origin.

### Frontend

```bash
cd frontend
VITE_API_URL=https://your-api-domain.com npm run build
# Serve the dist/ folder with nginx, Caddy, or a CDN.
```

Set `VITE_API_URL` to your backend's public URL so the built frontend points to the
correct API endpoint.

---

## Feature Classification

| Feature     | Values                                             |
|-------------|----------------------------------------------------|
| Domain      | `banking`, `supply chain`, `healthcare`, `general` |
| Complexity  | `low`, `medium`, `high`                            |
| Intent      | `list`, `compare`, `analyze`                       |
| Category    | `direct`, `generic`                                |
| Result_size | `small`, `medium`, `large`                         |
| Answer_type | `single_number`, `list`, `chart`, `table`           |

When `OPENAI_API_KEY` is set, the default OPENAI model classifies each question. Otherwise a
keyword-matching fallback is used.

---

## Input Token Pricing (per 1000000 tokens)

| Model | Price   |
|-------|---------|
| gpt-5.4 | $2.50   |

## Output Token Pricing (per 1000000 tokens)

| Model | Price  |
|-------|--------|
| gpt-5.4 | $15.00 |