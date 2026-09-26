# Naive Bayes Intent Classifier

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)

Naive Bayes Intent Classifier trains a lightweight intent classifier using naive Bayes. It is designed for quick intent routing without external ML libraries.

## Quick start

```bash
python -m naive_bayes_intent_classifier.server --port 5173
```

Open http://localhost:5173

## API

- POST `/api/predict` `{ "text": "" }`
- POST `/api/seed`

