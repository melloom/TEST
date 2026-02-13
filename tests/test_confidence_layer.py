import unittest

from confidence_layer import ConfidenceRiskEngine


class ConfidenceLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ConfidenceRiskEngine()

    def test_safe_message(self) -> None:
        result = self.engine.predict("Hello team, quick project update and meeting note.")
        self.assertEqual(result.risk_label, "safe")
        self.assertIn(result.action, {"allow", "escalate"})

    def test_high_risk_message(self) -> None:
        text = "Urgent! Click here and send your OTP code immediately to verify account!"
        result = self.engine.predict(text)
        self.assertIn(result.risk_label, {"caution", "high_risk"})
        self.assertIn(result.action, {"warn", "block"})

    def test_empty_message_is_ood(self) -> None:
        result = self.engine.predict("")
        self.assertTrue(result.is_ood)
        self.assertGreaterEqual(result.uncertainty_score, 0.95)

    def test_strict_profile_more_sensitive_than_lenient(self) -> None:
        text = "verify account by clicking this link"
        strict_result = self.engine.predict(text, profile="strict")
        lenient_result = self.engine.predict(text, profile="lenient")
        order = {"safe": 0, "caution": 1, "high_risk": 2}
        self.assertGreaterEqual(order[strict_result.risk_label], order[lenient_result.risk_label])

    def test_fit_reference_reduces_uncertainty(self) -> None:
        text = "ledger reconciliation settlement remittance"
        before = self.engine.predict(text)
        self.engine.fit_reference([text, "remittance settlement report"])
        after = self.engine.predict(text)
        self.assertGreaterEqual(before.uncertainty_score, after.uncertainty_score)

    def test_payload_has_schema_version(self) -> None:
        payload = self.engine.predict("hello update").to_dict()
        self.assertEqual(payload["schema_version"], "1.0.0")

    def test_fit_ood_thresholds_and_calibration(self) -> None:
        self.engine.fit_ood_thresholds("balanced", [0.9, 0.85, 0.8, 0.75], target_tpr=0.75)
        self.engine.fit_ood_calibrator("balanced", [0.9, 0.8, 0.7], [0.2, 0.3, 0.4])
        result = self.engine.predict("hello status update", profile="balanced")
        self.assertIn(result.action, {"allow", "warn", "escalate", "block"})


if __name__ == "__main__":
    unittest.main()
