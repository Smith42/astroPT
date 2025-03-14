import numpy as np
import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

class ModelEvaluator:
    def __init__(self, task_type="regression", parameter="PhotoZ"):
        self.task_type = task_type
        self.parameter = parameter

    def evaluate(self, model, embeddings, labels, n_mc_runs=10, poisson_noise=False):
        """
        Evaluate the model on the given data.

        Args:
            model: The trained model (PyTorch or scikit-learn).
            embeddings: Input features (numpy array or tensor).
            labels: True target values (numpy array).
            n_mc_runs: Number of Monte Carlo runs for evaluation.
            poisson_noise: Whether to add Poisson noise to predictions.

        Returns:
            final_metrics: Metrics computed on aggregated predictions.
            metrics_mean: Mean metrics across all runs.
            metrics_std: Standard deviation of metrics across all runs.
        """
        if isinstance(model, torch.nn.Module):
            model.eval()  # Set PyTorch model to evaluation mode
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
            model_mode = "pytorch"
        else:
            model_mode = "sklearn"

        # Initialize variables to store predictions and metrics across all Monte Carlo runs
        all_predictions = []
        all_metrics = []

        try:
            for _ in range(n_mc_runs):
                if model_mode == "pytorch":
                    with torch.no_grad():
                        predictions = model(embeddings_tensor).squeeze().numpy()
                else:
                    predictions = model.predict(embeddings)

                # Optionally add Poisson noise to the predictions
                if poisson_noise:
                    noise = np.sqrt(np.abs(predictions)) * np.random.normal(0, 0.1, size=predictions.shape)
                    predictions += noise

                all_predictions.append(predictions)

                # Compute metrics for this run
                if self.task_type == "regression":
                    metrics = self._compute_regression_metrics(labels, predictions)
                elif self.task_type == "binary_classification":
                    metrics = self._compute_classification_metrics(labels, predictions)
                else:
                    raise ValueError("Invalid task_type. Choose 'regression' or 'classification'.")

                all_metrics.append(metrics)

            # Aggregate predictions across all runs
            aggregated_predictions = np.mean(all_predictions, axis=0)

            # Compute final metrics on aggregated predictions
            if self.task_type == "regression":
                final_metrics = self._compute_regression_metrics(labels, aggregated_predictions)
            elif self.task_type == "binary_classification":
                final_metrics = self._compute_classification_metrics(labels, aggregated_predictions)

            # Compute mean and std of metrics across all runs
            metrics_mean = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
            metrics_std = {k: np.std([m[k] for m in all_metrics]) for k in all_metrics[0]}

            return final_metrics, metrics_mean, metrics_std, predictions

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None, None, None

    def _compute_regression_metrics(self, labels, predictions):
        """Compute regression metrics."""
        if self.parameter == 'PhotoZ':
            delta_z = (predictions - labels) / (1 + labels)
            outlier_fraction = 100 * len(predictions[np.abs(delta_z) >= 0.15]) / len(predictions)
        elif self.parameter == 'logM':
            delta_z = (predictions - labels) / (1 + labels)
            outlier_fraction = 100 * len(predictions[np.abs(delta_z) >= 0.25]) / len(predictions)
        else:
            print("The parameter is not defined")
        
        nmad = 1.48 * np.median(np.abs(delta_z - np.median(delta_z)))
        sigma_68 = (np.percentile(delta_z, 84.1) - np.percentile(delta_z, 15.9)) / 2
        

        return {
            "mse": mean_squared_error(labels, predictions),
            "mae": mean_absolute_error(labels, predictions),
            "r2": r2_score(labels, predictions),
            "bias": np.mean(delta_z),
            "nmad": nmad,
            "sigma_68": sigma_68,
            "outlier_fraction": outlier_fraction,
        }

    def _compute_classification_metrics(self, labels, predictions):
        """Compute classification metrics."""
        # Convert probabilities to binary predictions using a threshold of 0.5
        binary_predictions = (predictions >= 0.5).astype(int)
        
        # Compute metrics
        accuracy = accuracy_score(labels, binary_predictions)
        precision = precision_score(labels, binary_predictions, average="binary")
        recall = recall_score(labels, binary_predictions, average="binary")
        f1 = f1_score(labels, binary_predictions, average="binary")
        tn, fp, fn, tp = confusion_matrix(labels, binary_predictions).ravel()
        fpr = fp / (fp + tn)  # False Positive Rate

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
        }
