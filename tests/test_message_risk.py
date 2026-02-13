import unittest

from message_risk import MessageRiskClassifier, default_bootstrap_samples


class MessageRiskClassifierTests(unittest.TestCase):
    def test_classifier_bootstrap_ranks_risky_higher(self) -> None:
        clf = MessageRiskClassifier(embed_dim=64)
        clf.fit(default_bootstrap_samples(), lr=0.4, epochs=40)
        safe = clf.predict_score("hello team thanks for the project update")
        risky = clf.predict_score("urgent click here verify account and send otp")
        self.assertGreater(risky, safe)


if __name__ == "__main__":
    unittest.main()
