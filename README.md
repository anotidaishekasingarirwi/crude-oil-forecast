# WTI Crude Oil Price Forecast

Ensemble ML app (LSTM + CNN + LightGBM + Ridge) that forecasts WTI crude oil prices 10 days ahead.

## Project Structure

```
oil-forecast/
├── backend/
│   └── app.py              # Flask API
├── frontend/
│   └── index.html          # Dashboard UI
├── models/                 # Place your trained models here (not in git)
│   ├── lstm_model.h5
│   ├── cnn_model.h5
│   ├── lgb_model.pkl
│   ├── ridge_meta_model.pkl
│   ├── scaler.pkl
│   ├── config.json
│   └── metrics.json
├── requirements.txt
├── render.yaml
└── .gitignore
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/forecast/auto` | GET | Forecast using live Yahoo Finance data |
| `/api/forecast` | POST | Forecast using custom data |
| `/api/metrics` | GET | Model evaluation metrics |

## Local Development

```bash
pip install -r requirements.txt
# Place your models/ folder in the root
cd backend
python app.py
# Visit http://localhost:5000
```

## Deploy to Render

See DEPLOY.md for step-by-step instructions.
