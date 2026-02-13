# Confidence Layer Output Schema + Label Policy (v1)

Schema version: `1.0.0`

## Output schema

Each prediction MUST return:

- `schema_version` (`string`): schema contract version.
- `is_ood` (`boolean`): input appears out-of-distribution relative to reference baseline.
- `uncertainty_score` (`float`, `0.0..1.0`): confidence uncertainty where higher means less certainty.
- `risk_label` (`"safe" | "caution" | "high_risk"`): policy label.
- `risk_score` (`float`, `0.0..1.0`): scalar risk estimate.
- `reasons` (`string[]`): top factors supporting decision.
- `action` (`"allow" | "warn" | "block" | "escalate"`): product decision.
- `profile` (`"strict" | "balanced" | "lenient"`): threshold profile used.

## Risk scoring policy (fused)

- Rule signals detect explicit high-risk patterns (credentials, payment transfer, phishing CTA, PII exfiltration, link+account prompts).
- Classifier score estimates semantic risk from hashed embedding features.
- Final risk score is fused as: `0.6 * classifier_score + 0.4 * rule_score`.

## OoD / UE module

- OoD is produced by a dedicated `OodUeDetector` over calibrated in-domain score.
- Uncertainty is independently calibrated (`OodUeCalibrator`) and compared with `uncertainty_warn_threshold`.
- Each profile has explicit `ood_threshold` and `uncertainty_warn_threshold`.
- Threshold fitting uses lower quantile from in-domain calibration scores, keyed by target TPR.
- Safety hardening rules:
  - all score and threshold values are clamped into `[0, 1]`,
  - calibrator/threshold fitting gracefully no-ops when sample count is too small,
  - calibrator temperature is bounded to avoid extreme over/under calibration.

## Label policy

- `safe`: No strong abuse/phishing/exfiltration signal at configured profile threshold.
- `caution`: Potentially risky intent or uncertainty requiring warning/review path.
- `high_risk`: Strong risky intent indicators; should be blocked by default.

## Action policy

- `allow`: permit normal flow.
- `warn`: allow with visible warning and additional friction.
- `block`: deny action and capture audit event.
- `escalate`: route to fallback/human review when OOD with low direct risk signal.

## Threshold profiles

- `strict`: more sensitive (lower thresholds).
- `balanced`: default operating point.
- `lenient`: fewer false positives (higher thresholds).

## v1 decision matrix

- `high_risk` => `block`
- `caution` => `warn`
- `safe` + `is_ood=true` + `uncertainty_score > uncertainty_warn_threshold` => `warn`
- `safe` + `is_ood=true` + `uncertainty_score <= uncertainty_warn_threshold` => `escalate`
- `safe` + `is_ood=false` => `allow`

## Evaluation harness policy

- Evaluate with labeled examples containing `text`, `risk_label`, and `is_ood`.
- Primary dashboard metrics:
  - risk macro-F1,
  - OoD accuracy,
  - counts for risk FP/FN and OoD FP/FN.
- Error analysis output must include concrete FP/FN examples for threshold tuning.


## API + logging + versioned config policy

- API exposes both schema and config versions for reproducibility: `schema_version` and `config_version`.
- Runtime config endpoint (`GET /config`) provides active profile thresholds and embedder settings.
- Request logging should capture path, profile, outcome summary, and latency for operational debugging.

## Human-readable reason-string policy

- Reasons should be plain-language sentences understandable by non-ML users.
- Avoid purely technical shorthand in user-facing reasons.
- At least one reason is always returned; fallback reason is used when no strong signal is found.


## Production tuning policy

- Thresholds should be tuned on labeled validation sets using target ID TPR and high-risk precision targets.
- Use `tuning.py` to derive profile-specific `ood_threshold`, `uncertainty_warn_threshold`, `caution`, and `high_risk` values.
- Persist and audit tuned values through versioned config (`config_version`) and runtime config endpoint (`GET /config`).

## Reuse packaging policy

- Reuse the same confidence engine across message, mood, code, and assistant domains to keep one confidence contract.
- Domain wrappers may attach lightweight metadata (for example `domain`) but must preserve the core schema fields.
