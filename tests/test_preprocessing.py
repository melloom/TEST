import unittest

from preprocessing import HashingEmbedder, Preprocessor


class PreprocessingTests(unittest.TestCase):
    def test_url_normalization(self) -> None:
        p = Preprocessor()
        msg = p.process("Check https://example.com/login now")
        self.assertIn("<url>", msg.normalized)

    def test_embedder_dimension(self) -> None:
        e = HashingEmbedder(dim=64)
        vec = e.embed_tokens(["hello", "world"])
        self.assertEqual(len(vec), 64)
        self.assertAlmostEqual(sum(x * x for x in vec), 1.0, places=6)

    def test_embedder_is_deterministic(self) -> None:
        e = HashingEmbedder(dim=64)
        a = e.embed_tokens(["verify", "account", "now"])
        b = e.embed_tokens(["verify", "account", "now"])
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
