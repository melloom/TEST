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


if __name__ == "__main__":
    unittest.main()
