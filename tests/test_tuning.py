import unittest

from confidence_layer import ConfidenceRiskEngine
from tuning import ProductionThresholdTuner, TuningExample


class TuningTests(unittest.TestCase):
    def test_tune_profile_updates_thresholds(self) -> None:
        engine = ConfidenceRiskEngine()
        tuner = ProductionThresholdTuner(engine)
        before = engine.get_runtime_config()["profiles"]["balanced"]

        examples = [
            TuningExample(text="team sync status update", risk_label="safe", is_ood=False),
            TuningExample(text="urgent send otp now click here", risk_label="high_risk", is_ood=False),
            TuningExample(text="wire transfer gift card now", risk_label="high_risk", is_ood=False),
            TuningExample(text="zxqv jklp uiop", risk_label="safe", is_ood=True),
            TuningExample(text="qwerty asdf zxcv", risk_label="safe", is_ood=True),
        ]

        tuned = tuner.tune_profile("balanced", examples)
        after = engine.get_runtime_config()["profiles"]["balanced"]

        self.assertIn("high_risk", tuned)
        self.assertIn("ood_threshold", tuned)
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
