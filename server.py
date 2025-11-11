import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import argparse
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
import time
import json
from datetime import datetime
import threading
import sys

import numpy as np
import torch
import flwr as fl
from flwr.server.client_manager import SimpleClientManager, ClientProxy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.model_factory import get_model

RESULTS_BASE_DIR = os.path.abspath("Result/FLResult")
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

logger = logging.getLogger("FL-Server")

# (Optional) large message allowance for big models/metrics
GRPC_MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024  # 1GB

def get_init_parameters(model_name: str, num_classes: int) -> fl.common.Parameters:
    """Build the initial model and convert its state_dict to Flower Parameters."""
    try:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)

        model = get_model(model_name, num_classes, pretrained=False)

        with torch.no_grad():
            parameters = [v.cpu().numpy() for _, v in model.state_dict().items()]

        logger.info(f"Initial model parameters loaded for: {model_name}")
        logger.info("Server Model parameters shapes:")
        total_params = 0
        for name, param in model.state_dict().items():
            param_size = param.numel()
            logger.info(f"  {name}: shape={tuple(param.shape)}, size={param_size}")
            total_params += param_size
        logger.info(f"Total number of parameters: {total_params:,}")

        return fl.common.ndarrays_to_parameters(parameters)
    except Exception as e:
        logger.error(f"Failed to get initial parameters: {e}", exc_info=True)
        return None


def fit_config(server_round: int, local_epochs: int) -> Dict[str, fl.common.Scalar]:
    """Per-round training config broadcast to clients."""
    config = {
        "server_round": server_round,
        "local_epochs": local_epochs,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "loss_function": "cross_entropy",
        "optimizer": "adamw",
        "scheduler": "plateau",
        "use_scheduler": True,
        "batch_size": 16,
        "xai_probe": True,
        "xai_samples": 16,
        "xai_save_k": 0,
    }
    if server_round > 20:
        config["loss_function"] = "focal"
    if 32 <= server_round <= 60:
        config["learning_rate"] = 5e-4
        config["local_epochs"] = 4
    elif 61 <= server_round <= 80:
        config["learning_rate"] = 2e-4
        config["local_epochs"] = 3
    elif server_round > 80:
        config["learning_rate"] = 1e-4
        config["local_epochs"] = 2

    logger.info(
        f"Round {server_round} training config: "
        f"epochs={config['local_epochs']}, lr={config['learning_rate']}, loss={config['loss_function']}"
    )
    return config


def evaluate_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    return {"server_round": server_round}


def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Weighted average across client metrics dictionaries."""
    logger.info(f"Aggregating metrics from {len(metrics)} clients")
    if not metrics:
        return {}
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}
    aggregated_metrics: Dict[str, float] = {}
    for num_samples, client_metrics in metrics:
        weight = num_samples / total_samples
        for key, value in client_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + weight * float(value)
    return aggregated_metrics

class MedicalFLStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg strategy extended with:
      - history tracking
      - best/last global checkpoint saving
      - detailed round logging & plots
      - optional XAI metrics aggregation
      
    """
    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters: Optional[fl.common.Parameters] = None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
        results_base_dir: str = RESULTS_BASE_DIR,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.model_name = model_name
        self.num_classes = num_classes

        self.history = {
            "round": [],
            "train_loss": [],
            "train_accuracy": [],
            "train_f1": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "test_loss": [],
            "test_accuracy": [],
            "test_f1": [],
            "num_clients": [],
            "client_data_sizes": [],
            "aggregation_time": [],
            # ---- XAI history ----
            "xai_del_auc_mean": [], "xai_del_auc_std": [],
            "xai_heat_in_mask_mean": [], "xai_heat_in_mask_std": [],
        }
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
        self.best_round = 0
        self.best_parameters: Optional[fl.common.Parameters] = None
        self.last_parameters: Optional[fl.common.Parameters] = None

        self.connected_clients = set()
        self.client_metrics_history = {}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_base_dir = os.path.join(results_base_dir, f"fl_results_{ts}")
        os.makedirs(self.results_base_dir, exist_ok=True)
        self._save_strategy_config()

        logger.info("FL Strategy initialized")
        logger.info(f"   → results_base_dir: {self.results_base_dir}")
        logger.info(f"   → model={self.model_name}, num_classes={self.num_classes}")

    def _save_strategy_config(self):
        config = {
            "strategy": "MedicalFLStrategy",
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "fraction_fit": self.fraction_fit,
            "fraction_evaluate": self.fraction_evaluate,
            "min_fit_clients": self.min_fit_clients,
            "min_evaluate_clients": self.min_evaluate_clients,
            "min_available_clients": self.min_available_clients,
            "accept_failures": self.accept_failures,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(self.results_base_dir, "strategy_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        logger.info(f"Round {server_round}: configuring clients for training...")
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        self.connected_clients.update({c.cid for c in clients})

        logger.info(f"Selected {len(clients)} clients: {sorted([c.cid for c in clients])}")
        fit_ins = fl.common.FitIns(parameters, config)
        return [(c, fit_ins) for c in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []
        logger.info(f"Round {server_round}: configuring clients for evaluation...")
        config = self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {}
        sample_size, min_num = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        logger.info(f"Selected {len(clients)} clients for evaluation")

        eval_ins = fl.common.EvaluateIns(parameters, config)
        return [(c, eval_ins) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        t0 = time.time()
        logger.info(
            f"Round {server_round}: aggregating fit results (success={len(results)}, failures={len(failures)})"
        )

        if len(results) < self.min_fit_clients:
            logger.warning(
                f"Not enough results to aggregate. Expected {self.min_fit_clients}, got {len(results)}"
            )
            return None, {}

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is None:
            return None, aggregated_metrics

        self.last_parameters = aggregated_parameters

        # Summaries from client metrics
        summary = self._calculate_fit_metrics(results)
        self.history["round"].append(server_round)
        self.history["train_loss"].append(summary["train_loss_avg"])
        self.history["train_accuracy"].append(summary["train_accuracy_avg"])
        self.history["train_f1"].append(summary["train_f1_avg"])
        self.history["val_loss"].append(summary["val_loss_avg"])
        self.history["val_accuracy"].append(summary["val_accuracy_avg"])
        self.history["val_f1"].append(summary["val_f1_avg"])
        self.history["num_clients"].append(len(results))
        self.history["client_data_sizes"].append(summary["client_data_sizes"])
        self.history["aggregation_time"].append(time.time() - t0)

        # XAI history (only if keys exist in summary)
        self.history["xai_del_auc_mean"].append(summary.get("xai_del_auc_mean_avg", np.nan))
        self.history["xai_del_auc_std"].append(summary.get("xai_del_auc_std_avg", np.nan))
        self.history["xai_heat_in_mask_mean"].append(summary.get("xai_heat_in_mask_mean_avg", np.nan))
        self.history["xai_heat_in_mask_std"].append(summary.get("xai_heat_in_mask_std_avg", np.nan))

        # Track best by validation F1
        if summary["val_f1_avg"] > self.best_f1:
            self.best_f1 = summary["val_f1_avg"]
            self.best_accuracy = summary["val_accuracy_avg"]
            self.best_round = server_round
            self.best_parameters = aggregated_parameters
            self.save_best_model()
            logger.info(
                f"🏆 New best model: round={self.best_round}, val_f1={self.best_f1:.4f}, val_acc={self.best_accuracy:.4f}"
            )

        aggregated_metrics.update({k: v for k, v in summary.items() if k.startswith("xai_")})
        aggregated_metrics["aggregation_time"] = self.history["aggregation_time"][-1]
        self._log_round_summary(server_round, summary, len(results))

        # Periodic snapshot
        if server_round % 10 == 0:
            self.save_intermediate_results(server_round)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        logger.info(
            f"Round {server_round}: aggregating evaluation results (success={len(results)} fail={len(failures)})"
        )
        if not results:
            return None, {}
        test = self._calculate_eval_metrics(results)
        if len(self.history["test_loss"]) < len(self.history["round"]):
            self.history["test_loss"].append(test["test_loss_avg"])
            self.history["test_accuracy"].append(test["test_accuracy_avg"])
            self.history["test_f1"].append(test["test_f1_avg"])
        logger.info(
            f"   Test: loss={test['test_loss_avg']:.4f} acc={test['test_accuracy_avg']:.4f} f1={test['test_f1_avg']:.4f}"
        )
        return test["test_loss_avg"], test

    def _calculate_fit_metrics(self, results):
        total_examples = sum(fit_res.num_examples for _, fit_res in results) or 1
        metric_keys = [
            "train_loss", "train_accuracy", "train_f1",
            "val_loss", "val_accuracy", "val_f1",
            "xai_del_auc_mean", "xai_del_auc_std",
            "xai_heat_in_mask_mean", "xai_heat_in_mask_std",
        ]

        weighted_sums = {key: 0.0 for key in metric_keys}
        present = {key: False for key in metric_keys}
        client_data_sizes, client_metrics_list = [], []

        for client_proxy, fit_res in results:
            weight = fit_res.num_examples / total_examples
            client_data_sizes.append(fit_res.num_examples)
            metrics_for_client = {}
            for key in metric_keys:
                val = fit_res.metrics.get(key, None) if fit_res.metrics else None
                if isinstance(val, (int, float, np.integer, np.floating)):
                    weighted_sums[key] += float(val) * weight
                    present[key] = True
                    metrics_for_client[key] = float(val)
            client_metrics_list.append({
                "client_id": client_proxy.cid,
                "num_examples": fit_res.num_examples,
                "metrics": metrics_for_client,
            })

        current_round = len(self.history["round"]) + 1
        self.client_metrics_history[current_round] = client_metrics_list

        out = {
            "train_loss_avg": weighted_sums["train_loss"],
            "train_accuracy_avg": weighted_sums["train_accuracy"],
            "train_f1_avg": weighted_sums["train_f1"],
            "val_loss_avg": weighted_sums["val_loss"],
            "val_accuracy_avg": weighted_sums["val_accuracy"],
            "val_f1_avg": weighted_sums["val_f1"],
            "total_examples": total_examples,
            "client_data_sizes": client_data_sizes,
            "num_participating_clients": len(results),
        }
        if present["xai_del_auc_mean"]:
            out["xai_del_auc_mean_avg"] = weighted_sums["xai_del_auc_mean"]
            out["xai_del_auc_std_avg"]  = weighted_sums["xai_del_auc_std"]
        if present["xai_heat_in_mask_mean"]:
            out["xai_heat_in_mask_mean_avg"] = weighted_sums["xai_heat_in_mask_mean"]
            out["xai_heat_in_mask_std_avg"]  = weighted_sums["xai_heat_in_mask_std"]
        return out

    def _calculate_eval_metrics(
        self, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]]
    ) -> Dict:
        total_examples = sum(eval_res.num_examples for _, eval_res in results) or 1
        weighted_loss = 0.0
        weighted_accuracy = 0.0
        weighted_f1 = 0.0

        for _, eval_res in results:
            weight = eval_res.num_examples / total_examples
            weighted_loss += float(eval_res.loss or 0.0) * weight
            if eval_res.metrics:
                weighted_accuracy += float(eval_res.metrics.get("accuracy", 0.0)) * weight
                weighted_f1 += float(eval_res.metrics.get("f1_macro", 0.0)) * weight

        return {
            "test_loss_avg": weighted_loss,
            "test_accuracy_avg": weighted_accuracy,
            "test_f1_avg": weighted_f1,
            "total_test_examples": total_examples,
            "num_eval_clients": len(results),
        }

    def _log_round_summary(self, round_num: int, summary: Dict, num_clients: int):
        logger.info("=" * 80)
        logger.info(f"ROUND {round_num} SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Clients: {num_clients} | examples: {summary['total_examples']:,}")
        logger.info(
            f"Train: loss={summary['train_loss_avg']:.4f} "
            f"acc={summary['train_accuracy_avg']:.4f} f1={summary['train_f1_avg']:.4f}"
        )
        logger.info(
            f"Val  : loss={summary['val_loss_avg']:.4f} "
            f"acc={summary['val_accuracy_avg']:.4f} f1={summary['val_f1_avg']:.4f}"
        )
        logger.info(f"Best : round={self.best_round} val_f1={self.best_f1:.4f} val_acc={self.best_accuracy:.4f}")

        xai_mean = summary.get("xai_del_auc_mean_avg", None)
        if xai_mean is not None:
            logger.info(
                f"XAI  : delAUC={xai_mean:.4f} "
                f"mask_in={summary.get('xai_heat_in_mask_mean_avg','NA')}"
            )

        logger.info("=" * 80)

    def save_best_model(self) -> None:
        if self.best_parameters is None:
            logger.warning("No best parameters available, skipping model save.")
            return
        try:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)

            model = get_model(self.model_name, num_classes=self.num_classes, pretrained=False)

            best_state_dict = OrderedDict()
            for (name, param), arr in zip(
                model.state_dict().items(),
                fl.common.parameters_to_ndarrays(self.best_parameters),
            ):
                best_state_dict[name] = torch.as_tensor(arr, dtype=param.dtype)

            checkpoint = {
                "round": self.best_round,
                "model_state_dict": best_state_dict,
                "best_f1": self.best_f1,
                "best_accuracy": self.best_accuracy,
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "timestamp": datetime.now().isoformat(),
            }

            save_path = os.path.join(self.results_base_dir, f"best_model_round_{self.best_round}.pth")
            torch.save(checkpoint, save_path)
            logger.info(f"Best model saved successfully → {save_path}")

        except Exception as exc:
            logger.error(f"Failed to save best model: {exc}", exc_info=True)

    def save_last_model(self) -> None:
        if self.last_parameters is None:
            logger.warning("No last parameters available, skipping model save.")
            return
        try:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)

            model = get_model(self.model_name, num_classes=self.num_classes, pretrained=False)

            last_state_dict = OrderedDict()
            for (name, param), arr in zip(
                model.state_dict().items(),
                fl.common.parameters_to_ndarrays(self.last_parameters),
            ):
                last_state_dict[name] = torch.as_tensor(arr, dtype=param.dtype)

            checkpoint = {
                "model_state_dict": last_state_dict,
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "timestamp": datetime.now().isoformat(),
            }

            save_path = os.path.join(self.results_base_dir, "last_global_model.pth")
            torch.save(checkpoint, save_path)
            logger.info(f"Last global model saved successfully → {save_path}")

        except Exception as exc:
            logger.error(f"Failed to save last model: {exc}", exc_info=True)

    def save_intermediate_results(self, rnd: int):
        try:
            with open(os.path.join(self.results_base_dir, f"history_round_{rnd}.json"), "w") as f:
                json.dump(self.history, f, indent=2)
            with open(os.path.join(self.results_base_dir, f"client_metrics_round_{rnd}.json"), "w") as f:
                json.dump(self.client_metrics_history, f, indent=2)
            self.plot_training_curves(save_suffix=f"_round_{rnd}")
            logger.info(f"Intermediate results saved for round {rnd}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}", exc_info=True)

    def save_final_results(self):
        try:
            with open(os.path.join(self.results_base_dir, "final_training_history.json"), "w") as f:
                json.dump(self.history, f, indent=2)
            with open(os.path.join(self.results_base_dir, "final_client_metrics.json"), "w") as f:
                json.dump(self.client_metrics_history, f, indent=2)
            self.plot_training_curves(save_suffix="_final")
            logger.info(f"Final results saved in {self.results_base_dir}")
        except Exception as e:
            logger.error(f"Failed to save final results: {e}", exc_info=True)

    # def plot_training_curves(self, save_suffix: str = ""):
        # if not self.history["round"]:
        #     return
        # rounds = self.history["round"]
        # fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # # Loss
        # axes[0, 0].plot(rounds, self.history["train_loss"], label="Train Loss", linewidth=2, marker="o")
        # axes[0, 0].plot(rounds, self.history["val_loss"], label="Val Loss", linewidth=2, marker="s")
        # if self.history["test_loss"]:
        #     # FIX: Slice the rounds to match the number of available test loss points
        #     num_test_points = len(self.history["test_loss"])
        #     axes[0, 0].plot(rounds[:num_test_points], self.history["test_loss"], label="Test Loss", linewidth=2, marker="^")
        # axes[0, 0].set_title("Loss")

        # # Acc
        # axes[0, 1].plot(rounds, self.history["train_accuracy"], label="Train Acc", linewidth=2, marker="o")
        # axes[0, 1].plot(rounds, self.history["val_accuracy"], label="Val Acc", linewidth=2, marker="s")
        # if self.history["test_accuracy"]:
        #     # FIX: Slice the rounds to match the number of available test accuracy points
        #     num_test_points = len(self.history["test_accuracy"])
        #     axes[0, 1].plot(rounds[:num_test_points], self.history["test_accuracy"], label="Test Acc", linewidth=2, marker="^")
        # axes[0, 1].set_title("Accuracy")

        # # F1
        # axes[0, 2].plot(rounds, self.history["train_f1"], label="Train F1", linewidth=2, marker="o")
        # axes[0, 2].plot(rounds, self.history["val_f1"], label="Val F1", linewidth=2, marker="s")
        # if self.history["test_f1"]:
        #     # FIX: Slice the rounds to match the number of available test F1 points
        #     num_test_points = len(self.history["test_f1"])
        #     axes[0, 2].plot(rounds[:num_test_points], self.history["test_f1"], label="Test F1", linewidth=2, marker="^")
        # axes[0, 2].set_title("F1-Score")

        # # Clients per round
        # axes[1, 0].bar(rounds, self.history["num_clients"], alpha=0.8, label="Clients")
        # axes[1, 0].set_title("Clients per Round")

        # # Aggregation time
        # if self.history["aggregation_time"]:
        #     axes[1, 1].plot(rounds, self.history["aggregation_time"], linewidth=2, marker="d", label="Agg Time (s)")
        #     axes[1, 1].set_title("Aggregation Time (s)")

        # # Data distribution (latest round)
        # if self.history["client_data_sizes"]:
        #     latest = self.history["client_data_sizes"][-1]
        #     labels = [f"C{i+1}" for i in range(len(latest))]
        #     axes[1, 2].pie(latest, labels=labels, autopct="%1.1f%%", startangle=90)
        #     axes[1, 2].set_title("Data Distribution (latest)")

        # for ax in axes.ravel():
        #     ax.grid(True, alpha=0.3)
        #     handles, labels = ax.get_legend_handles_labels()
        #     if handles:
        #         ax.legend(loc="best")

        # plt.tight_layout()
        # out = os.path.join(self.results_base_dir, f"training_curves{save_suffix}.png")
        # plt.savefig(out, dpi=300, bbox_inches="tight")
        # plt.close()
        # logger.info(f"Saved plot → {out}")

    def plot_training_curves(self, save_suffix: str = "", theme: str = "dark", custom_colors: dict | None = None):
        """
        Save two figures with a themed color palette.

        Parameters
        ----------
        save_suffix : str
            Suffix for output filenames.
        theme : {"dark","light"}
            Selects rcParam bundle + default color palette.
        custom_colors : dict | None
            Optional overrides, e.g. {
                "train": "#4C78A8",
                "val":   "#F58518",
                "test":  "#54A24B",
                "agg":   "#B279A2",
                "bar":   "#4C78A8",
                "pie":   ["#4C78A8","#F58518","#54A24B","#E45756","#72B7B2","#EECA3B"]
            }
        """
        import os
        import matplotlib.pyplot as plt
        from matplotlib import rc_context

        if not self.history.get("round") or not self.history["round"]:
            return

        # ---------- THEMES ----------
        THEMES = {
            "dark": {
                "rc": {
                    "figure.facecolor": "white",
                    "axes.facecolor":   "white",
                    "axes.edgecolor":   "#222222",
                    "axes.labelcolor":  "#222222",
                    "xtick.color":      "#222222",
                    "ytick.color":      "#222222",
                    "grid.color":       "#CCCCCC",
                    "text.color":       "#222222",
                    "legend.facecolor": "white",
                    "legend.edgecolor": "#DDDDDD",
                },
                # Okabe–Ito palette (color-blind friendly) + a couple extras
                "colors": {
                    "train": "#56B4E9",  # sky blue
                    "val":   "#E69F00",  # orange
                    "test":  "#009E73",  # bluish green
                    "agg":   "#CC79A7",  # reddish purple
                    "bar":   "#56B4E9",
                    "pie":   ["#56B4E9","#E69F00","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7"]
                }
            },
            "light": {
                "rc": {
                    "figure.facecolor": "white",
                    "axes.facecolor":   "white",
                    "axes.edgecolor":   "#222222",
                    "axes.labelcolor":  "#222222",
                    "xtick.color":      "#222222",
                    "ytick.color":      "#222222",
                    "grid.color":       "#CCCCCC",
                    "text.color":       "#222222",
                    "legend.facecolor": "white",
                    "legend.edgecolor": "#DDDDDD",
                },
                "colors": {
                    "train": "#1f77b4",  # blue
                    "val":   "#ff7f0e",  # orange
                    "test":  "#2ca02c",  # green
                    "agg":   "#9467bd",  # purple
                    "bar":   "#1f77b4",
                    "pie":   ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#17becf","#bcbd22","#9467bd"]
                }
            }
        }

        # Select theme & merge custom overrides
        theme = theme if theme in THEMES else "dark"
        rc_bundle = THEMES[theme]["rc"].copy()
        color_bundle = THEMES[theme]["colors"].copy()

        if custom_colors:
            for k, v in custom_colors.items():
                if k == "pie" and isinstance(v, (list, tuple)):
                    color_bundle["pie"] = list(v)
                else:
                    color_bundle[k] = v

        rounds = list(self.history["round"])

        def _plot_pair(ax, x, y_train, y_val, y_test, title, yl=None):
            if y_train:
                ax.plot(x[:len(y_train)], y_train, label="Train", linewidth=2, marker="o", color=color_bundle["train"])
            if y_val:
                ax.plot(x[:len(y_val)], y_val, label="Val", linewidth=2, marker="s", color=color_bundle["val"])
            if y_test:
                ax.plot(x[:len(y_test)], y_test, label="Test", linewidth=2, marker="^", color=color_bundle["test"])
            ax.set_title(title)
            if yl: ax.set_ylabel(yl)
            ax.grid(True, alpha=0.35)
            h, _ = ax.get_legend_handles_labels()
            if h: ax.legend(loc="best")

        with rc_context(rc=rc_bundle):
            # ---------- Figure 1: Core metrics ----------
            fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

            _plot_pair(
                axes1[0, 0],
                rounds,
                self.history.get("train_loss", []),
                self.history.get("val_loss", []),
                self.history.get("test_loss", []),
                "Loss", yl="Loss"
            )

            _plot_pair(
                axes1[0, 1],
                rounds,
                self.history.get("train_accuracy", []),
                self.history.get("val_accuracy", []),
                self.history.get("test_accuracy", []),
                "Accuracy", yl="Accuracy"
            )

            _plot_pair(
                axes1[1, 0],
                rounds,
                self.history.get("train_f1", []),
                self.history.get("val_f1", []),
                self.history.get("test_f1", []),
                "F1-Score", yl="F1"
            )

            ax_ag = axes1[1, 1]
            agg_time = self.history.get("aggregation_time", [])
            if agg_time:
                ax_ag.plot(rounds[:len(agg_time)], agg_time, linewidth=2, marker="d",
                        label="Agg Time (s)", color=color_bundle["agg"])
            ax_ag.set_title("Aggregation Time (s)")
            ax_ag.set_ylabel("Seconds")
            ax_ag.grid(True, alpha=0.35)
            h, _ = ax_ag.get_legend_handles_labels()
            if h: ax_ag.legend(loc="best")

            plt.tight_layout()
            out1 = os.path.join(self.results_base_dir, f"training_core_metrics{save_suffix}.png")
            plt.savefig(out1, dpi=300, bbox_inches="tight")
            plt.close(fig1)

            # ---------- Figure 2: Rounds & Distribution ----------
            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

            # Clients per round (bar)
            num_clients = self.history.get("num_clients", [])
            ax_bar = axes2[0]
            if num_clients:
                ax_bar.bar(rounds[:len(num_clients)], num_clients, alpha=0.9, label="Clients", color=color_bundle["bar"])
            ax_bar.set_title("Clients per Round")
            ax_bar.set_xlabel("Round")
            ax_bar.set_ylabel("Clients")
            ax_bar.grid(True, axis="y", alpha=0.3)
            h, _ = ax_bar.get_legend_handles_labels()
            if h: ax_bar.legend(loc="best")

            # Latest data distribution (pie)
            ax_pie = axes2[1]
            client_sizes_hist = self.history.get("client_data_sizes", [])
            if client_sizes_hist and client_sizes_hist[-1]:
                latest = client_sizes_hist[-1]
                total = sum(latest) if latest else 0
                if total > 0:
                    labels = [f"C{i+1}" for i in range(len(latest))]
                    pie_colors = (color_bundle.get("pie") or [None])[:len(latest)]
                    ax_pie.pie(latest, labels=labels, autopct="%1.1f%%", startangle=90, colors=pie_colors)
                    ax_pie.set_aspect("equal")
            ax_pie.set_title("Data Distribution (latest)")

            plt.tight_layout()
            out2 = os.path.join(self.results_base_dir, f"rounds_and_distribution{save_suffix}.png")
            plt.savefig(out2, dpi=300, bbox_inches="tight")
            plt.close(fig2)

        logger.info(f"Saved plots → {out1} and {out2}")


class LoggingClientManager(SimpleClientManager):
    def __init__(self, expected_clients: int):
        super().__init__()
        self.expected_clients = expected_clients

    def register(self, client: ClientProxy) -> bool:
        ok = super().register(client)
        n = self.num_available()
        remaining = max(self.expected_clients - n, 0)
        logger.info(f"Client connected: {client.cid} | connected={n} | waiting={remaining}")
        if n >= self.expected_clients:
            logger.info("Required clients connected. Starting rounds as soon as strategy is ready.")
        return ok

    def unregister(self, client: ClientProxy) -> None:
        super().unregister(client)
        n = self.num_available()
        remaining = max(self.expected_clients - n, 0)
        logger.info(f"Client disconnected: {client.cid} | connected={n} | waiting={remaining}")

def start_waiting_heartbeat(cm: SimpleClientManager, target: int, interval_sec: float = 2.0):
    stop_evt = threading.Event()

    def _loop():
        while not stop_evt.is_set():
            connected = cm.num_available()
            remaining = max(target - connected, 0)
            if remaining <= 0:
                stop_evt.set()
                break
            logger.info(f"⏳ Waiting for clients… connected={connected} | waiting={remaining}")
            time.sleep(interval_sec)

    thr = threading.Thread(target=_loop, daemon=True)
    thr.start()
    return stop_evt

def create_server_strategy(
    *,
    min_clients: int,
    fraction_fit: float,
    fraction_evaluate: float,
    model_name: str = "customcnn",
    num_classes: int,
    local_epochs: int,
) -> MedicalFLStrategy:
    initial_parameters = get_init_parameters(model_name, num_classes)
    if initial_parameters is None:
        raise RuntimeError("Failed to initialize model parameters")

    return MedicalFLStrategy(
        model_name=model_name,
        num_classes=num_classes,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_clients,
        min_evaluate_clients=1,
        min_available_clients=min_clients,
        on_fit_config_fn=lambda server_round: fit_config(server_round, local_epochs),
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )


def main():
    parser = argparse.ArgumentParser("Federated Learning Server (Medical)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8080, help="Port")
    parser.add_argument("--rounds", type=int, default=3, help="FL rounds")
    parser.add_argument("--min-clients", type=int, default=1, help="Minimum clients per round")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients to train each round")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients to evaluate each round")
    parser.add_argument("--model", type=str, default="customcnn",
                        choices=["mobilenetv3", "hybridmodel", "resnet50", "customcnn", "hybridswin", "densenet121"])
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--local-epochs", type=int, default=6, help="Local epochs per round")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--expected-clients", type=int, default=None,
                        help="How many clients you expect to connect (for logs). Defaults to --min-clients.")
    args = parser.parse_args()

    if args.expected_clients is None:
        args.expected_clients = args.min_clients

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,
    )

    logger.info("Starting Federated Learning Server")
    logger.info("=" * 80)
    logger.info(f"Host={args.host}  Port={args.port}  Rounds={args.rounds}")
    logger.info(f"Model={args.model}  NumClasses={args.num_classes}")
    logger.info(f"MinClients={args.min_clients}  FitFrac={args.fraction_fit}  EvalFrac={args.fraction_evaluate}")
    logger.info(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    logger.info("=" * 80)

    try:
        strategy = create_server_strategy(
            min_clients=args.min_clients,
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            model_name=args.model,
            num_classes=args.num_classes,
            local_epochs=args.local_epochs,
        )
        client_manager = LoggingClientManager(expected_clients=args.expected_clients)

        server_cfg = fl.server.ServerConfig(num_rounds=args.rounds)
        server_addr = f"{args.host}:{args.port}"

        logger.info(f"🌐 Flower gRPC server → {server_addr}")
        logger.info(f"🎯 Expecting {args.expected_clients} clients to connect")

        # (Optional) Start heartbeat log while waiting for clients
        hb_stop = start_waiting_heartbeat(client_manager, args.expected_clients, interval_sec=2.0)

        # --- NEW: supported args only ---
        fl.server.start_server(
            server_address=server_addr,
            config=server_cfg,
            strategy=strategy,
            client_manager=client_manager,
            grpc_max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        )

        try:
            hb_stop.set()
        except Exception:
            pass

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        try:
            # Persist results
            strategy.save_final_results()
            strategy.save_last_model()
            if strategy.history["round"]:
                logger.info("\nExperiment finished")
                logger.info(f"Results: {strategy.results_base_dir}")
                logger.info(
                    f"Best round: {strategy.best_round}  "
                    f"val_f1={strategy.best_f1:.4f}  val_acc={strategy.best_accuracy:.4f}"
                )
            else:
                logger.warning("No rounds completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()