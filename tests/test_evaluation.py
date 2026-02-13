import unittest

from evaluation import EvalExample, EvaluationHarness


class EvaluationHarnessTests(unittest.TestCase):
    def test_evaluation_outputs_dashboard_and_error_buckets(self) -> None:
        harness = EvaluationHarness()
        examples = [
            EvalExample(text="team standup at 10", risk_label="safe", is_ood=False),
            EvalExample(text="urgent send your password now", risk_label="high_risk", is_ood=False),
            EvalExample(text="asdf qwer zxcv", risk_label="safe", is_ood=True),
        ]

        report = harness.evaluate(examples, profile="balanced")
        self.assertIn("dashboard_markdown", report)
        self.assertIn("error_analysis", report)
        self.assertGreaterEqual(report["risk_macro_f1"], 0.0)
        self.assertGreaterEqual(report["ood_accuracy"], 0.0)

        dashboard = report["dashboard_markdown"]
        self.assertIn("Confidence Layer Evaluation Dashboard", dashboard)

        errors = report["error_analysis"]
        self.assertIn("risk_false_positives", errors)
        self.assertIn("risk_false_negatives", errors)
        self.assertIn("ood_false_positives", errors)
        self.assertIn("ood_false_negatives", errors)


if __name__ == "__main__":
    unittest.main()
