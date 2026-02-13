# TEST

Confidence layer v1 with:
- finalized output schema and label/action policy
- preprocessing pipeline (normalization + tokenization)
- deterministic dependency-free embedding pipeline (stable hashed embeddings)
- dedicated OoD/UE module with calibration + threshold fitting
- OoD + uncertainty + message-risk prediction

## Files

- `schema.py`: canonical response contract and typed prediction payload.
- `label_policy.md`: label/action policy and threshold profile definitions.
- `preprocessing.py`: text preprocessing and embedding pipeline.
- `ood_ue.py`: dedicated OoD/UE detector, calibrator, and threshold fitting utilities.
- `confidence_layer.py`: main inference engine.
- `api.py`: HTTP server exposing prediction endpoint.

## Run API

```bash
python api.py
```

Server endpoints:

- `GET /health`
- `GET /schema`
- `POST /predict-confidence-risk`

Example request:

```bash
curl -s -X POST http://localhost:8080/predict-confidence-risk \
  -H 'Content-Type: application/json' \
  -d '{"text":"Urgent! Click here and send your OTP","profile":"strict"}'
```

## Calibration + thresholds

```python
from confidence_layer import ConfidenceRiskEngine

engine = ConfidenceRiskEngine()
engine.fit_ood_thresholds("balanced", [0.92, 0.88, 0.85, 0.79], target_tpr=0.95)
engine.fit_ood_calibrator("balanced", [0.9, 0.85, 0.8], [0.25, 0.30, 0.35])
```

## Run tests

```bash
python -m unittest discover -s tests -v
```
