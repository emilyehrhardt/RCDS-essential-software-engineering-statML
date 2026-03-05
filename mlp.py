"""Simple NumPy implementation of a Multi Layer Perceptron (MLP)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class MultiLayerPerceptron:
    """Fully-connected neural network trained with mini-batch gradient descent.

    Supported tasks:
    - ``binary_classification``
    - ``multiclass_classification``
    - ``regression``
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: Sequence[int] = (16,),
        output_dim: int = 1,
        task: str = "binary_classification",
        hidden_activation: str = "relu",
        learning_rate: float = 0.01,
        l2: float = 0.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.output_dim = output_dim
        self.task = task
        self.hidden_activation = hidden_activation
        self.learning_rate = learning_rate
        self.l2 = l2
        self.random_state = random_state

        self._validate_configuration()
        self._rng = np.random.default_rng(self.random_state)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._initialize_parameters()

    def _validate_configuration(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be a positive integer")
        if any(layer_size <= 0 for layer_size in self.hidden_layer_sizes):
            raise ValueError("All hidden layer sizes must be positive integers")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.l2 < 0:
            raise ValueError("l2 regularization must be >= 0")

        valid_tasks = {"binary_classification", "multiclass_classification", "regression"}
        if self.task not in valid_tasks:
            raise ValueError(f"Unsupported task '{self.task}'")

        if self.task == "binary_classification" and self.output_dim != 1:
            raise ValueError("binary_classification requires output_dim == 1")
        if self.task == "multiclass_classification" and self.output_dim < 2:
            raise ValueError("multiclass_classification requires output_dim >= 2")

        valid_hidden_activations = {"relu", "tanh"}
        if self.hidden_activation not in valid_hidden_activations:
            raise ValueError(
                f"Unsupported hidden_activation '{self.hidden_activation}'"
            )

    def _initialize_parameters(self) -> None:
        layer_dims = [self.input_dim, *self.hidden_layer_sizes, self.output_dim]
        self.weights = []
        self.biases = []

        for idx in range(len(layer_dims) - 1):
            fan_in = layer_dims[idx]
            fan_out = layer_dims[idx + 1]

            if idx < len(self.hidden_layer_sizes) and self.hidden_activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)

            self.weights.append(self._rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=float))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_derivative(z: np.ndarray) -> np.ndarray:
        return (z > 0.0).astype(float)

    @staticmethod
    def _tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def _tanh_derivative(z: np.ndarray) -> np.ndarray:
        tanh_z = np.tanh(z)
        return 1.0 - tanh_z * tanh_z

    def _hidden_forward(self, z: np.ndarray) -> np.ndarray:
        if self.hidden_activation == "relu":
            return self._relu(z)
        return self._tanh(z)

    def _hidden_backward(self, z: np.ndarray) -> np.ndarray:
        if self.hidden_activation == "relu":
            return self._relu_derivative(z)
        return self._tanh_derivative(z)

    def _output_forward(self, z: np.ndarray) -> np.ndarray:
        if self.task == "binary_classification":
            return self._sigmoid(z)
        if self.task == "multiclass_classification":
            return self._softmax(z)
        return z

    def _check_features(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("X must be a 2D array")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"X has {x.shape[1]} features, expected {self.input_dim}"
            )
        return x

    def _prepare_targets(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.shape[0] == 0:
            raise ValueError("y cannot be empty")

        if self.task == "binary_classification":
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if y.ndim != 2 or y.shape[1] != 1:
                raise ValueError("For binary classification, y must have shape (n,) or (n, 1)")
            return y.astype(float)

        if self.task == "multiclass_classification":
            if y.ndim == 1:
                if np.any((y < 0) | (y >= self.output_dim)):
                    raise ValueError("Class labels in y are out of range")
                one_hot = np.zeros((y.shape[0], self.output_dim), dtype=float)
                one_hot[np.arange(y.shape[0]), y.astype(int)] = 1.0
                return one_hot
            if y.ndim == 2 and y.shape[1] == self.output_dim:
                return y.astype(float)
            raise ValueError(
                f"For multiclass classification, y must have shape (n,) or (n, {self.output_dim})"
            )

        if y.ndim == 1:
            if self.output_dim != 1:
                raise ValueError(
                    f"For regression with output_dim={self.output_dim}, y must be 2D"
                )
            y = y.reshape(-1, 1)
        if y.ndim != 2 or y.shape[1] != self.output_dim:
            raise ValueError(
                f"For regression, y must have shape (n, {self.output_dim})"
            )
        return y.astype(float)

    def _forward_pass(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [x]
        pre_activations = []
        current = x

        for layer_idx in range(len(self.weights) - 1):
            z = current @ self.weights[layer_idx] + self.biases[layer_idx]
            pre_activations.append(z)
            current = self._hidden_forward(z)
            activations.append(current)

        z_out = current @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z_out)
        out = self._output_forward(z_out)
        activations.append(out)
        return activations, pre_activations

    def _loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-12
        if self.task == "binary_classification":
            clipped = np.clip(y_pred, eps, 1.0 - eps)
            return float(-np.mean(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped)))
        if self.task == "multiclass_classification":
            clipped = np.clip(y_pred, eps, 1.0 - eps)
            return float(-np.mean(np.sum(y_true * np.log(clipped), axis=1)))
        return float(np.mean((y_true - y_pred) ** 2))

    def _backward_pass(
        self,
        activations: List[np.ndarray],
        pre_activations: List[np.ndarray],
        y_true: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        m = y_true.shape[0]
        grad_w: List[np.ndarray] = [np.empty_like(weight) for weight in self.weights]
        grad_b: List[np.ndarray] = [np.empty_like(bias) for bias in self.biases]

        y_pred = activations[-1]
        if self.task == "regression":
            delta = (2.0 / m) * (y_pred - y_true)
        else:
            delta = (y_pred - y_true) / m

        for layer_idx in reversed(range(len(self.weights))):
            a_prev = activations[layer_idx]
            grad_w[layer_idx] = a_prev.T @ delta + (self.l2 / m) * self.weights[layer_idx]
            grad_b[layer_idx] = np.sum(delta, axis=0, keepdims=True)

            if layer_idx > 0:
                delta = (delta @ self.weights[layer_idx].T) * self._hidden_backward(
                    pre_activations[layer_idx - 1]
                )

        return grad_w, grad_b

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 500,
        batch_size: Optional[int] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = False,
    ) -> Dict[str, List[float]]:
        """Train the model and return loss history."""
        if epochs <= 0:
            raise ValueError("epochs must be a positive integer")

        x = self._check_features(x)
        y = self._prepare_targets(y)

        if x.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        if batch_size is None:
            batch_size = x.shape[0]
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        x_val = None
        y_val = None
        if validation_data is not None:
            x_val_raw, y_val_raw = validation_data
            x_val = self._check_features(x_val_raw)
            y_val = self._prepare_targets(y_val_raw)

        history: Dict[str, List[float]] = {"loss": []}
        if validation_data is not None:
            history["val_loss"] = []

        for epoch in range(epochs):
            indices = self._rng.permutation(x.shape[0])

            for start in range(0, x.shape[0], batch_size):
                batch_idx = indices[start : start + batch_size]
                x_batch = x[batch_idx]
                y_batch = y[batch_idx]

                activations, pre_activations = self._forward_pass(x_batch)
                grad_w, grad_b = self._backward_pass(activations, pre_activations, y_batch)

                for idx in range(len(self.weights)):
                    self.weights[idx] -= self.learning_rate * grad_w[idx]
                    self.biases[idx] -= self.learning_rate * grad_b[idx]

            train_loss = self._loss(y, self._forward_pass(x)[0][-1])
            history["loss"].append(train_loss)

            if validation_data is not None and x_val is not None and y_val is not None:
                val_loss = self._loss(y_val, self._forward_pass(x_val)[0][-1])
                history["val_loss"].append(val_loss)

            if verbose and ((epoch + 1) == 1 or (epoch + 1) % 100 == 0):
                msg = f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.6f}"
                if validation_data is not None and "val_loss" in history:
                    msg += f" - val_loss: {history['val_loss'][-1]:.6f}"
                print(msg)

        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict probabilities for classification tasks."""
        if self.task == "regression":
            raise ValueError("predict_proba is only available for classification tasks")
        x = self._check_features(x)
        output = self._forward_pass(x)[0][-1]
        if self.task == "binary_classification":
            return output.ravel()
        return output

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict classes (classification) or numeric outputs (regression)."""
        x = self._check_features(x)
        output = self._forward_pass(x)[0][-1]

        if self.task == "binary_classification":
            return (output.ravel() >= threshold).astype(int)
        if self.task == "multiclass_classification":
            return np.argmax(output, axis=1)
        if self.output_dim == 1:
            return output.ravel()
        return output

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on a dataset."""
        x = self._check_features(x)
        y_true = self._prepare_targets(y)
        y_pred = self._forward_pass(x)[0][-1]
        results: Dict[str, float] = {"loss": self._loss(y_true, y_pred)}

        if self.task == "binary_classification":
            y_labels = y_true.ravel().astype(int)
            y_hat = (y_pred.ravel() >= 0.5).astype(int)
            results["accuracy"] = float(np.mean(y_labels == y_hat))
        elif self.task == "multiclass_classification":
            y_labels = np.argmax(y_true, axis=1)
            y_hat = np.argmax(y_pred, axis=1)
            results["accuracy"] = float(np.mean(y_labels == y_hat))

        return results
