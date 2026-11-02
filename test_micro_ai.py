import json
from pathlib import Path
import tempfile
import unittest
from micro_ai import Model, pairs, softmax


class ModelTests(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        model = Model("ab")
        batch = pairs("aabab", model)
        original = [row[:] for row in model.weights]
        gradient = model.step(batch, 0.1)
        model.weights = original
        epsilon = 1e-5
        for i in range(2):
            for j in range(2):
                model.weights[i][j] += epsilon
                high = model.loss(batch)
                model.weights[i][j] -= 2 * epsilon
                low = model.loss(batch)
                model.weights[i][j] += epsilon
                self.assertAlmostEqual(gradient[i][j], (high-low)/(2*epsilon), places=7)

    def test_training_learns_alternation(self):
        model = Model("ab")
        batch = pairs("ab" * 20, model)
        before = model.loss(batch)
        for _ in range(100):
            model.step(batch, 1)
        self.assertLess(model.loss(batch), before / 10)
        self.assertEqual(model.generate("a", 20, temperature=0.1), "ab" * 10 + "a")

    def test_stability_roundtrip_and_corrupt_checkpoint(self):
        self.assertEqual(softmax([10000, 10000]), [0.5, 0.5])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            model = Model("ab")
            model.save(path)
            self.assertEqual(Model.load(path).generate("a"), model.generate("a"))
            with self.assertRaises(FileExistsError):
                model.save(path)
            path.write_text(json.dumps({"version": 1, "alphabet": "ab", "weights": [[1]]}))
            with self.assertRaises(ValueError):
                Model.load(path)

    def test_invalid_generation(self):
        for prompt, length, temperature in [("", 10, 1), ("x", 10, 1), ("a", -1, 1), ("a", 10, 0)]:
            with self.assertRaises((ValueError, KeyError)):
                Model("ab").generate(prompt, length, temperature=temperature)

    def test_checkpoint_root_and_parameter_budgets_fail_cleanly(self):
        for alphabet in [None, [], "a"*257, "aa"]:
            with self.assertRaises(ValueError): Model(alphabet)
        for length in [True, 1.5, "2"]:
            with self.assertRaises(ValueError): Model("ab").generate("a", length)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.json"
            for raw in ["[]", "null", '"text"', " "*4_000_001]:
                path.write_text(raw)
                with self.assertRaises(ValueError): Model.load(path)


if __name__ == "__main__":
    unittest.main()
