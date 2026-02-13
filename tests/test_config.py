import unittest

from config import CONFIG_VERSION, default_config


class ConfigTests(unittest.TestCase):
    def test_default_config_is_versioned(self) -> None:
        cfg = default_config()
        self.assertEqual(cfg.config_version, CONFIG_VERSION)
        self.assertIn("balanced", cfg.profiles)
        self.assertGreater(cfg.embed_dim, 0)


if __name__ == "__main__":
    unittest.main()
