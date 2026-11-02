import math
import unittest
import json
import tempfile
from pathlib import Path
from context_model import ContextModel


class ContextTests(unittest.TestCase):
    def test_all_parameter_gradients_match_finite_differences(self):
        model = ContextModel("ab", context=2, hidden=3)
        examples = model.examples("aabbaab")
        _, gradients = model.loss_and_gradients(examples)
        for name, matrix in model.parameters.items():
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    epsilon = 1e-5
                    row[j] = value + epsilon
                    high = model.loss(examples)
                    row[j] = value - epsilon
                    low = model.loss(examples)
                    row[j] = value
                    self.assertAlmostEqual(gradients[name][i][j], (high-low)/(2*epsilon), places=7)

    def test_network_learns_context_that_bigram_cannot_disambiguate(self):
        model = ContextModel("ab", context=2, hidden=8)
        examples = model.examples("aabb" * 12)
        before = model.loss(examples)
        for _ in range(350):
            model.step(examples, 1.0)
        self.assertLess(model.loss(examples), before/10)
        self.assertEqual(model.generate("aa", 14, temperature=0.1), "aabb"*4)

    def test_generation_is_reproducible_and_invalid_inputs_are_rejected(self):
        model = ContextModel("ab", context=2)
        self.assertEqual(model.generate("aa"), model.generate("aa"))
        for context in [[], [-1, 0], [0, 2], [True, 0]]:
            with self.assertRaises(ValueError): model.forward(context)
        for rate in [0, -1, math.nan, math.inf]:
            with self.assertRaises(ValueError): model.step(model.examples("aab"), rate)
        with self.assertRaises(ValueError): model.loss([])
        with self.assertRaises(ValueError): model.generate("a")

    def test_checkpoint_roundtrip_refuses_overwrites_and_malformed_parameters(self):
        model = ContextModel("ab", context=2, hidden=3)
        model.step(model.examples("aabbab"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.json"
            model.save(path)
            self.assertEqual(ContextModel.load(path).generate("aa"), model.generate("aa"))
            with self.assertRaises(FileExistsError): model.save(path)
            data = json.loads(path.read_text())
            for invalid in [float("nan"), float("inf"), True, 1001, "exec()"]:
                data["parameters"]["input"][0][0] = invalid
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError): ContextModel.load(path)
            for raw in ["[]", "{}", '"unexpected"', " "*4_000_001]:
                path.write_text(raw)
                with self.assertRaises(ValueError): ContextModel.load(path)


if __name__ == "__main__":
    unittest.main()
