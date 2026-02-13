import unittest

from ood_ue import OodUeCalibrator, OodUeDetector, OodUeThresholds


class OodUeTests(unittest.TestCase):
    def test_detector_threshold(self) -> None:
        detector = OodUeDetector(thresholds=OodUeThresholds(ood_threshold=0.8, uncertainty_warn_threshold=0.75))
        res = detector.evaluate(0.5, 0.8)
        self.assertTrue(res.is_ood)

    def test_fit_thresholds(self) -> None:
        detector = OodUeDetector()
        detector.fit_thresholds([0.9, 0.8, 0.7, 0.6], target_tpr=0.75)
        self.assertGreaterEqual(detector.thresholds.ood_threshold, 0.6)

    def test_calibrator_fit(self) -> None:
        cal = OodUeCalibrator.fit_from_scores([0.9, 0.85], [0.2, 0.3])
        id_score = cal.calibrate_in_domain(0.9)
        ood_score = cal.calibrate_in_domain(0.2)
        self.assertGreater(id_score, ood_score)


if __name__ == "__main__":
    unittest.main()
