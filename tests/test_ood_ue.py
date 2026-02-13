import unittest

from ood_ue import OodUeCalibrator, OodUeDetector, OodUeThresholds


class OodUeTests(unittest.TestCase):
    def test_detector_threshold(self) -> None:
        detector = OodUeDetector(thresholds=OodUeThresholds(ood_threshold=0.8, uncertainty_warn_threshold=0.75))
        res = detector.evaluate(0.5, 0.8)
        self.assertTrue(res.is_ood)

    def test_fit_thresholds(self) -> None:
        detector = OodUeDetector()
        detector.fit_thresholds([0.9, 0.8, 0.7, 0.6] * 3, target_tpr=0.75)
        self.assertGreaterEqual(detector.thresholds.ood_threshold, 0.6)
        self.assertLessEqual(detector.thresholds.ood_threshold, 0.9)

    def test_fit_thresholds_skips_when_not_enough_samples(self) -> None:
        detector = OodUeDetector()
        old = detector.thresholds.ood_threshold
        detector.fit_thresholds([0.8, 0.7], target_tpr=0.9, min_samples=5)
        self.assertEqual(old, detector.thresholds.ood_threshold)

    def test_calibrator_fit(self) -> None:
        cal = OodUeCalibrator.fit_from_scores([0.9, 0.85], [0.2, 0.3])
        id_score = cal.calibrate_in_domain(0.9)
        ood_score = cal.calibrate_in_domain(0.2)
        self.assertGreater(id_score, ood_score)

    def test_thresholds_and_scores_are_clamped(self) -> None:
        detector = OodUeDetector(thresholds=OodUeThresholds(ood_threshold=2.0, uncertainty_warn_threshold=-1.0))
        self.assertEqual(detector.thresholds.ood_threshold, 1.0)
        self.assertEqual(detector.thresholds.uncertainty_warn_threshold, 0.0)

        res = detector.evaluate(3.0, -4.0)
        self.assertGreaterEqual(res.in_domain_score, 0.0)
        self.assertLessEqual(res.in_domain_score, 1.0)
        self.assertGreaterEqual(res.uncertainty_score, 0.0)
        self.assertLessEqual(res.uncertainty_score, 1.0)


if __name__ == "__main__":
    unittest.main()
