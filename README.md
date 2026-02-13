# TEST

v0 confidence layer scaffold for:
- OoD detection (`is_ood`)
- uncertainty estimation (`uncertainty_score`)
- message-risk prediction (`risk_label`, `risk_score`)

## Run API

```bash
python api.py
```

Server endpoint:

- `GET /health`
- `POST /predict-confidence-risk`

Example request:

```bash
curl -s -X POST http://localhost:8080/predict-confidence-risk \
  -H 'Content-Type: application/json' \
  -d '{"text":"Urgent! Click here and send your OTP"}' | jq
```

## Run tests

```bash
python -m unittest discover -s tests -v
```
