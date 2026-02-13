# TEST

Confidence layer v1 with:
- finalized output schema and label/action policy
- preprocessing pipeline (normalization + tokenization)
- deterministic dependency-free embedding pipeline (stable hashed embeddings)
- dedicated OoD/UE module with calibration + threshold fitting
- message-risk classifier + rule signals fused into one risk score
- evaluation harness with dashboard summary and FP/FN error analysis
- API logging + versioned runtime configuration
- human-readable reason strings for model decisions

## Files

- `schema.py`: canonical response contract and typed prediction payload.
- `config.py`: versioned engine config with profile thresholds and risk-rule catalog.
- `label_policy.md`: label/action policy and threshold profile definitions.
- `preprocessing.py`: text preprocessing and embedding pipeline.
- `ood_ue.py`: dedicated OoD/UE detector, calibrator, and threshold fitting utilities.
- `message_risk.py`: lightweight message-risk classifier and bootstrap trainer.
- `confidence_layer.py`: main inference engine with fused risk scoring.
- `evaluation.py`: evaluation harness, dashboard output, and error bucket analysis.
- `api.py`: HTTP server exposing prediction and evaluation endpoints.

## Run API

```bash
python api.py
```

Server endpoints:

- `GET /health`
- `GET /schema`
- `GET /config`
- `POST /predict-confidence-risk`
- `POST /evaluate`

`/schema` includes both `schema_version` and `config_version`.
`/config` returns active profile thresholds and embedder dimension for runtime traceability.

## Logging

Request-level logs are emitted with:
- endpoint path
- selected profile
- prediction summary (`risk_label`, `is_ood`) for inference requests
- sample count for evaluation requests
- request duration in milliseconds

## Calibration + thresholds (hardened defaults)

```python
from confidence_layer import ConfidenceRiskEngine

engine = ConfidenceRiskEngine()

# threshold fitting requires enough in-domain samples (min_samples default=10)
engine.fit_ood_thresholds("balanced", [0.92, 0.88, 0.85, 0.79, 0.81, 0.83, 0.86, 0.84, 0.82, 0.8], target_tpr=0.95)

# calibrator fitting falls back to safe defaults when sample sizes are too small
engine.fit_ood_calibrator("balanced", [0.9, 0.85, 0.8], [0.25, 0.30, 0.35])
```

## Evaluation harness + dashboard

```python
from evaluation import EvalExample, EvaluationHarness

harness = EvaluationHarness()
report = harness.evaluate(
    [
        EvalExample(text="team update by noon", risk_label="safe", is_ood=False),
        EvalExample(text="urgent click here and verify account", risk_label="high_risk", is_ood=False),
        EvalExample(text="asdf qwer zxcv", risk_label="safe", is_ood=True),
    ],
    profile="balanced",
)

print(report["dashboard_markdown"])
print(report["error_analysis"])
```

## Run tests

```bash
python -m unittest discover -s tests -v
```
