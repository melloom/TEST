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

## OoD / UE module

- OoD is produced by a dedicated `OodUeDetector` over calibrated in-domain score.
- Uncertainty is independently calibrated (`OodUeCalibrator`) and compared with `uncertainty_warn_threshold`.
- Each profile has explicit `ood_threshold` and `uncertainty_warn_threshold`.
- Thresholds can be fit with ID scores using target TPR.

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
