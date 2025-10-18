import os
import flwr as fl
import torch
from project.models import SepsisNet  # replace with your actual model
from flwr.common import parameters_to_ndarrays

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

NUM_CLIENTS = 3  # you can change this
NUM_ROUNDS = 3

# --------------------------
# Aggregation functions
# --------------------------
def weighted_average(metrics_list):
    # metrics_list is now a list of tuples: (num_examples, metrics_dict)
    total_examples = sum(num_examples for num_examples, _ in metrics_list)
    if total_examples == 0:
        return {}

    # Weighted average of accuracy, loss, etc.
    aggregated_metrics = {}
    for num_examples, metrics in metrics_list:
        for key, value in metrics.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0.0
            aggregated_metrics[key] += value * num_examples

    for key in aggregated_metrics:
        aggregated_metrics[key] /= total_examples

    return aggregated_metrics

# --------------------------
# Custom FedAvg strategy with logging
# --------------------------
class LoggingFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is None:
            return None
        parameters, num_examples = aggregated
        try:
            ndarrays = parameters_to_ndarrays(parameters)
            model = SepsisNet()
            keys = list(model.state_dict().keys())
            state_dict = {k: torch.tensor(p) for k, p in zip(keys, ndarrays)}
            save_path = os.path.join(LOG_DIR, f"global_model_round{rnd}.pth")
            torch.save(state_dict, save_path)
            print(f"[SERVER] Saved global model round {rnd} -> {save_path}")
        except Exception as e:
            print("Warning: could not save global model state:", e)
        return aggregated

# --------------------------
# Strategy setup
# --------------------------
strategy = LoggingFedAvg(
    fraction_fit=1.0,
    min_fit_clients=1,
    min_available_clients=1,
    evaluate_metrics_aggregation_fn=weighted_average,  # <--- fix for nan summary
    fit_metrics_aggregation_fn=weighted_average        # optional: aggregate fit metrics if reported
)

# --------------------------
# Start server
# --------------------------
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)
