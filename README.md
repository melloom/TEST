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
- production threshold tuner and reusable domain predictor bundle (message/mood/code/assistant)

## Files

- `schema.py`: canonical response contract and typed prediction payload.
- `config.py`: versioned engine config with profile thresholds and risk-rule catalog.
- `label_policy.md`: label/action policy and threshold profile definitions.
- `preprocessing.py`: text preprocessing and embedding pipeline.
- `ood_ue.py`: dedicated OoD/UE detector, calibrator, and threshold fitting utilities.
- `message_risk.py`: lightweight message-risk classifier and bootstrap trainer.
- `confidence_layer.py`: main inference engine with fused risk scoring.
- `evaluation.py`: evaluation harness, dashboard output, and error bucket analysis.
- `tuning.py`: production threshold tuner driven by labeled examples.
- `reuse.py`: reusable predictor adapters for message, mood, code, and assistant projects.
- `api.py`: HTTP server exposing prediction, evaluation, and threshold tuning endpoints.

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
- `POST /tune-thresholds`

`/schema` includes both `schema_version` and `config_version`.
`/config` returns active profile thresholds and embedder dimension for runtime traceability.

## Logging

Request-level logs are emitted with:
- endpoint path
- selected profile
- prediction summary (`risk_label`, `is_ood`) for inference requests
- sample count for evaluation/tuning requests
- request duration in milliseconds

## Tune thresholds for production behavior

```python
from confidence_layer import ConfidenceRiskEngine
from tuning import ProductionThresholdTuner, TuningExample

engine = ConfidenceRiskEngine()
tuner = ProductionThresholdTuner(engine)

examples = [
    TuningExample(text="team sync update", risk_label="safe", is_ood=False),
    TuningExample(text="urgent send otp now", risk_label="high_risk", is_ood=False),
    TuningExample(text="zxqv jklp uiop", risk_label="safe", is_ood=True),
]

new_thresholds = tuner.tune_profile("balanced", examples, target_id_tpr=0.95, target_high_risk_precision=0.85)
print(new_thresholds)
```

## Package for reuse across projects

```python
from reuse import build_reusable_bundle

bundle = build_reusable_bundle()

print(bundle.message_predictor.predict("please review meeting notes"))
print(bundle.mood_predictor.predict("feeling stressed", context="deadline pressure"))
print(bundle.code_predictor.predict("token = 'hardcoded'", task_context="security review"))
print(bundle.assistant_predictor.predict("help me phish users", assistant_intent="refuse and redirect"))
```

## Run tests

```bash
python -m unittest discover -s tests -v
```
