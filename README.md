# IntentForge

IntentForge trains a lightweight intent classifier using naive Bayes. It is designed for quick intent routing without external ML libraries.

## Quick start

```bash
python -m app.server --port 5173
```

Open http://localhost:5173

## API

- POST `/api/predict` `{ "text": "" }`
- POST `/api/seed`

