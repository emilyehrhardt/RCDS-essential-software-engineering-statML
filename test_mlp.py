import unittest

import numpy as np

from mlp import MultiLayerPerceptron


class TestMultiLayerPerceptron(unittest.TestCase):
    def test_binary_classification_xor(self):
        x = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        y = np.array([0, 1, 1, 0])

        model = MultiLayerPerceptron(
            input_dim=2,
            hidden_layer_sizes=(6,),
            output_dim=1,
            task="binary_classification",
            hidden_activation="tanh",
            learning_rate=0.2,
            random_state=7,
        )
        history = model.fit(x, y, epochs=4000, batch_size=4)

        self.assertLess(history["loss"][-1], history["loss"][0])
        np.testing.assert_array_equal(model.predict(x), y)

    def test_multiclass_classification(self):
        rng = np.random.default_rng(123)
        x0 = rng.normal(loc=(-2.0, 0.0), scale=0.25, size=(40, 2))
        x1 = rng.normal(loc=(2.0, 0.0), scale=0.25, size=(40, 2))
        x2 = rng.normal(loc=(0.0, 2.5), scale=0.25, size=(40, 2))
        x = np.vstack([x0, x1, x2])
        y = np.array([0] * 40 + [1] * 40 + [2] * 40)

        model = MultiLayerPerceptron(
            input_dim=2,
            hidden_layer_sizes=(12,),
            output_dim=3,
            task="multiclass_classification",
            learning_rate=0.05,
            random_state=1,
        )
        model.fit(x, y, epochs=1200, batch_size=32)
        metrics = model.evaluate(x, y)

        self.assertGreater(metrics["accuracy"], 0.95)

    def test_regression(self):
        x = np.linspace(-2.0, 2.0, 120).reshape(-1, 1)
        y = 3.0 * x.ravel() - 0.5

        model = MultiLayerPerceptron(
            input_dim=1,
            hidden_layer_sizes=(),
            output_dim=1,
            task="regression",
            learning_rate=0.05,
            random_state=0,
        )
        history = model.fit(x, y, epochs=800, batch_size=24)
        predictions = model.predict(x)

        mse = float(np.mean((predictions - y) ** 2))
        self.assertLess(history["loss"][-1], history["loss"][0])
        self.assertLess(mse, 1e-3)


if __name__ == "__main__":
    unittest.main()
