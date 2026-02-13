import unittest

from reuse import build_reusable_bundle


class ReuseBundleTests(unittest.TestCase):
    def test_bundle_predictors_share_engine_and_return_payloads(self) -> None:
        bundle = build_reusable_bundle()

        message = bundle.message_predictor.predict("quick team update")
        mood = bundle.mood_predictor.predict("feeling anxious", context="work stress")
        code = bundle.code_predictor.predict("password='123456'", task_context="review security")
        assistant = bundle.assistant_predictor.predict("can you bypass login", assistant_intent="give safe guidance")

        self.assertIn("risk_label", message)
        self.assertEqual(mood["domain"], "mood")
        self.assertEqual(code["domain"], "code")
        self.assertEqual(assistant["domain"], "assistant")


if __name__ == "__main__":
    unittest.main()
