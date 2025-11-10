import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# gRPC stability / payloads
os.environ.setdefault("GRPC_KEEPALIVE_TIME_MS", "30000")
os.environ.setdefault("GRPC_KEEPALIVE_TIMEOUT_MS", "10000")
os.environ.setdefault("GRPC_HTTP2_MAX_PINGS_WITHOUT_DATA", "0")
os.environ.setdefault("GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS", "1")
os.environ.setdefault("GRPC_MAX_RECEIVE_MESSAGE_LENGTH", str(200 * 1024 * 1024))
os.environ.setdefault("GRPC_MAX_SEND_MESSAGE_LENGTH",    str(200 * 1024 * 1024))

import logging, warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import multiprocessing as mp
import time
import argparse
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
import grpc
import cv2

from models.model_factory import get_model, FocalLoss, LabelSmoothingLoss
from utils.dataloder import create_data_loaders, get_class_weights  # (spelling kept)
from utils.train_eval import ModelTrainer, ModelMetrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

RESULTS_BASE_DIR = os.path.abspath(os.path.join("Result", "clientresult"))
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)


# Small CPU/runtime knobs
def _set_runtime_knobs(num_threads: int = 4) -> None:
    """
    Configure CPU threads to reduce contention on CPU-only boxes.
    """
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    try:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


# XAI helpers (Grad-CAM)
def _normalize01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    amin = a.min()
    a = a - amin
    amax = a.max() + 1e-12
    a = a / amax
    return a

def _find_last_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

class _GradCAM:
    """Minimal, fast Grad-CAM for single-image probe."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.tl = target_layer
        self.A = None
        self.dA = None
        self.ha = self.tl.register_forward_hook(self._hook_act)
        self.hg = self.tl.register_full_backward_hook(self._hook_grad)

    def _hook_act(self, module, inp, out):
        self.A = out

    def _hook_grad(self, module, gin, gout):
        self.dA = gout[0]

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        A = self.A[0]               # [C,H,W]
        dA = self.dA[0]             # [C,H,W]
        w = dA.mean(dim=(1, 2))     # [C]
        cam = torch.relu((w[:, None, None] * A).sum(dim=0)).detach().cpu().numpy()
        return _normalize01(cam)

    def close(self):
        self.ha.remove()
        self.hg.remove()

def _overlay_on_gray(img_u8: np.ndarray, heat: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """img_u8: HxW uint8; heat: HxW [0..1]; returns HxWx3 BGR (OpenCV)."""
    H, W = img_u8.shape
    heat_r = cv2.resize(heat, (W, H))
    heatmap = cv2.applyColorMap((heat_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 1.0, heatmap, alpha, 0)

def _deletion_curve_scores(model: nn.Module, x: torch.Tensor, heat: np.ndarray, steps: int = 10) -> List[float]:
    """Iteratively zero most-important pixels; record target logit."""
    device = next(model.parameters()).device
    x = x.clone().to(device)
    with torch.no_grad():
        base_logits = model(x)[0]
        cls = int(base_logits.argmax().item())
        scores = [base_logits[cls].item()]

    H, W = heat.shape
    order = np.argsort(-heat.flatten())  # descending by importance
    k = int(np.ceil(len(order) / steps))
    for s in range(steps):
        idxs = order[s * k:(s + 1) * k]
        for idx in idxs:
            y, z = idx // W, idx % W
            x[0, 0, y, z] = 0.0
        with torch.no_grad():
            scores.append(model(x)[0, cls].item())
    return scores

def _auc_trapz(y: List[float]) -> float:
    y = np.asarray(y, dtype=np.float32)
    if y.size < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, y.size, dtype=np.float32)
    # NumPy >=1.20 has trapezoid; this name is stable in 1.26+
    return float(np.trapezoid(y, x))


# Flower Client
class MedicalFLClient(fl.client.NumPyClient):
    """
    Federated Learning client for medical image classification (PyTorch).
    """
    def __init__(
        self,
        client_id: int,
        data_dir: str,
        device: torch.device,
        model_name: str = "customcnn",
        num_classes: int = 3,
        batch_size: int = 16,
        local_epochs: int = 8,
        num_workers: int = 4,
        results_base_dir: str = RESULTS_BASE_DIR,
    ):
        self.client_id = client_id
        self.data_dir = data_dir
        self.device = device
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.num_workers = num_workers

        # Per-client folders
        self.results_base_dir = results_base_dir
        os.makedirs(self.results_base_dir, exist_ok=True)
        self.client_root = os.path.join(self.results_base_dir, f"client_{client_id}")
        self.ckpt_dir = os.path.join(self.client_root, "checkpoints")
        self.log_dir = os.path.join(self.client_root, "logs")
        self.xai_dir = os.path.join(self.client_root, "xai")
        self.pred_dir = os.path.join(self.client_root, "predictions")
        self.metrics_dir = os.path.join(self.client_root, "metrics")
        for d in [self.client_root, self.ckpt_dir, self.log_dir, self.xai_dir, self.pred_dir, self.metrics_dir]:
            os.makedirs(d, exist_ok=True)

        # Model
        self.model = get_model(model_name, num_classes, pretrained=True)
        self.model.to(device)

        # Choose a conv layer for Grad-CAM (optional)
        self.target_layer = _find_last_conv(self.model)
        if self.target_layer is None:
            logger.warning("No Conv2d layer found for Grad-CAM. XAI probe will be skipped.")

        # DataLoaders (with spawn-safe multiprocessing context)
        logger.info(f"Client {client_id}: Loading data from {data_dir}")
        ctx = mp.get_context("spawn")
        self.train_loader, self.val_loader, self.test_loader = self._create_loaders_spawn_safe(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=(224, 224),
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            num_workers=num_workers,
            multiprocessing_context=ctx,
        )

        # Class weights
        self.class_weights = get_class_weights(self.train_loader)
        try:
            cw_log = self.class_weights.tolist()
        except Exception:
            cw_log = self.class_weights
        logger.info(f"Client {client_id}: Class weights: {cw_log}")

        # Trainer
        self.trainer = ModelTrainer(self.model, device, self.ckpt_dir, self.log_dir)

        # Defaults (can be overridden by server config)
        self.learning_rate = 0.001
        self.weight_decay = 1e-4

        logger.info(f"Client {client_id} initialized successfully")
        logger.info(f"  - Training samples:   {len(self.train_loader.dataset)}")
        logger.info(f"  - Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"  - Test samples:       {len(self.test_loader.dataset)}")

    @staticmethod
    def _create_loaders_spawn_safe(
        data_dir: str,
        batch_size: int,
        image_size: Tuple[int, int],
        train_split: float,
        val_split: float,
        test_split: float,
        num_workers: int,
        multiprocessing_context,
    ):
        """
        Try to pass 'multiprocessing_context' to your create_data_loaders if supported.
        If not, fall back without it (still fine if mp.set_start_method('spawn') is set).
        """
        try:
            loaders = create_data_loaders(
                data_dir=data_dir,
                batch_size=batch_size,
                train_split=train_split,
                val_split=val_split,
                test_split=test_split,
                image_size=image_size,
                num_workers=num_workers,
                multiprocessing_context=multiprocessing_context,  # may raise TypeError if not supported
            )
            return loaders
        except TypeError:
            logger.warning("utils.dataloder.create_data_loaders does not accept 'multiprocessing_context'; "
                           "falling back without it. Make sure mp.set_start_method('spawn') is set in main().")
            loaders = create_data_loaders(
                data_dir=data_dir,
                batch_size=batch_size,
                train_split=train_split,
                val_split=val_split,
                test_split=test_split,
                image_size=image_size,
                num_workers=num_workers,
            )
            return loaders

    # Flower NumPyClient API ----
    def get_parameters(self, config: Dict = None) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        own_state = self.model.state_dict()
        keys = list(own_state.keys())
        incoming = OrderedDict()
        for k, v in zip(keys, parameters):
            t = torch.tensor(v, dtype=own_state[k].dtype)
            incoming[k] = t
        merged = OrderedDict((k, incoming.get(k, own_state[k])) for k in own_state.keys())
        self.model.load_state_dict(merged, strict=False)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        logger.info(f"Client {self.client_id}: Starting local training round")
        self.set_parameters(parameters)

        # Server-configurable knobs
        local_epochs   = int(config.get("local_epochs", self.local_epochs))
        learning_rate  = float(config.get("learning_rate", self.learning_rate))
        weight_decay   = float(config.get("weight_decay", self.weight_decay))
        loss_function  = str(config.get("loss_function", "crossentropy")).lower()
        optimizer_name = str(config.get("optimizer", "adamw")).lower()
        scheduler_name = str(config.get("scheduler", "plateau")).lower()
        use_scheduler  = bool(config.get("use_scheduler", True))

        # Loss
        if loss_function == "focal":
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif loss_function == "label_smoothing":
            criterion = LabelSmoothingLoss(num_classes=self.num_classes, smoothing=0.1)
        else:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))

        # Train
        train_history = self.trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=local_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            class_weights=self.class_weights,
            use_scheduler=use_scheduler,
            patience=10,
            criterion=criterion,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
        )

        # Evaluate on test
        test_metrics = self.trainer.evaluate(self.test_loader)

        # XAI probe (optional)
        xai_metrics = self._xai_probe(self.val_loader, num_samples=16, save_k=3) if self.target_layer else {
            "xai_del_auc_mean": 0.0, "xai_del_auc_std": 0.0
        }

        # Save checkpoint
        best_model_path = os.path.join(self.ckpt_dir, f"client_{self.client_id}_best_model.pth")
        torch.save(self.model.state_dict(), best_model_path)
        logger.info(f"Client {self.client_id}: Best model saved to {best_model_path}")

        # Scalar metrics only (keep payload small)
        metrics = {
            "train_loss": float(train_history["train_loss"][-1]),
            "train_accuracy": float(train_history["train_accuracy"][-1]),
            "val_loss": float(train_history["val_loss"][-1]),
            "val_accuracy": float(train_history["val_accuracy"][-1]),
            "val_f1": float(train_history["val_f1_macro"][-1]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_f1": float(test_metrics["f1_macro"]),
            "num_examples": int(len(self.train_loader.dataset)),
            **xai_metrics,
        }

        logger.info(f"Client {self.client_id}: Local training completed")
        return self.get_parameters(), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        logger.info(f"Client {self.client_id}: Starting evaluation")
        self.set_parameters(parameters)

        test_metrics = self.trainer.evaluate(self.test_loader)
        add_xai = self._xai_probe(self.val_loader, num_samples=12, save_k=0) if self.target_layer else {
            "xai_del_auc_mean": 0.0, "xai_del_auc_std": 0.0
        }
        test_metrics.update(add_xai)

        logger.info(f"Client {self.client_id}: Evaluation completed")
        logger.info(f"  - Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  - Test F1 (Macro): {test_metrics['f1_macro']:.4f}")

        return (
            float(test_metrics.get("loss", 0.0)),
            int(len(self.test_loader.dataset)),
            {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in test_metrics.items()},
        )

    def _xai_probe(self, loader, num_samples: int = 16, save_k: int = 3) -> Dict:
        if self.target_layer is None:
            return {"xai_del_auc_mean": 0.0, "xai_del_auc_std": 0.0}

        self.model.eval()
        device = self.device
        cam_engine = _GradCAM(self.model, self.target_layer)

        del_aucs, saved, seen = [], 0, 0
        for data, _ in loader:
            B = data.size(0)
            for i in range(B):
                x = data[i:i+1].to(device)  # [1, C, H, W]

                # Predict class without grads
                with torch.no_grad():
                    logits = self.model(x)
                    pred_idx = int(torch.argmax(logits, dim=1).item())

                # CAM with grads
                with torch.enable_grad():
                    x = x.requires_grad_(True)
                    heat = cam_engine.generate(x, class_idx=pred_idx)  # [H,W], [0..1]

                # Faithfulness metric (deletion AUC)
                scores = _deletion_curve_scores(self.model, x.detach(), heat, steps=10)
                del_aucs.append(_auc_trapz(scores))

                # Save a few overlays
                if save_k and saved < save_k:
                    img = data[i, 0].cpu().numpy()
                    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
                    img_u8 = (img * 255).astype(np.uint8)
                    overlay_bgr = _overlay_on_gray(img_u8, heat, alpha=0.35)
                    name = f"round_overlay_{saved + 1}.png"
                    out_path = os.path.join(self.xai_dir, name)
                    cv2.imwrite(out_path, overlay_bgr)
                    saved += 1

                seen += 1
                if seen >= num_samples:
                    break
            if seen >= num_samples:
                break

        cam_engine.close()

        if not del_aucs:
            return {"xai_del_auc_mean": 0.0, "xai_del_auc_std": 0.0}
        return {
            "xai_del_auc_mean": float(np.mean(del_aucs)),
            "xai_del_auc_std": float(np.std(del_aucs)),
        }


def create_client(client_id: int, data_dir: str, model_name: str = "customcnn",
                  batch_size: int = 16, local_epochs: int = 50, num_workers: int = 3) -> MedicalFLClient:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    client = MedicalFLClient(
        client_id=client_id,
        data_dir=data_dir,
        device=device,
        model_name=model_name,
        num_classes=3,
        batch_size=batch_size,
        local_epochs=local_epochs,
        num_workers=num_workers,
    )
    return client


def run_flower(server_address: str, client: MedicalFLClient) -> None:
    """Start Flower with simple auto-reconnect on transient UNAVAILABLE."""
    while True:
        try:
            fl.client.start_client(
                server_address=server_address,
                client=client.to_client()
            )
            break  # finished cleanly
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                logger.warning("Server UNAVAILABLE (gRPC 14). Reconnecting in 5s...")
                time.sleep(5)
                continue
            else:
                raise


def main():
    # Use spawn to avoid forking a multi-threaded process (Flower/gRPC)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set in this process
        pass

    _set_runtime_knobs(num_threads=4)

    parser = argparse.ArgumentParser(description="Federated Learning Client for Medical Imaging")
    parser.add_argument("--client-id", type=int, default=1, help="Client ID")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to client data directory")
    parser.add_argument("--server-address", type=str, default="localhost:8080", help="FL server address")
    parser.add_argument("--model", type=str, default="customcnn",
                        choices=["mobilenetv3", "hybridmodel", "resnet50", "customcnn", "hybridswin", "densenet121"],
                        help="Model architecture")
    parser.add_argument("--train-local", action="store_true", help="Run local training only (no FL server)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (CPU-only: 32 for ResNet/DenseNet, 64 for small CNN/EfficientNetB0)")
    parser.add_argument("--local-epochs", type=int, default=50, help="Local epochs per round")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (set 0 if problems)")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory not found: {args.data_dir}")

    client = create_client(
        client_id=args.client_id,
        data_dir=args.data_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        num_workers=args.num_workers,
    )

    if args.train_local:
        logger.info("Running standalone local training (no FL server)")
        updated_params, num_examples, train_metrics = client.fit(client.get_parameters(), config={})
        test_loss, test_examples, test_metrics = client.evaluate(updated_params, config={})
        logger.info("Local training and evaluation completed:")
        logger.info(f"  - Final train metrics: {train_metrics}")
        logger.info(f"  - Final test metrics:  {test_metrics}")
        return

    logger.info(f"Starting FL client {args.client_id} connecting to {args.server_address}")
    run_flower(args.server_address, client)


if __name__ == "__main__":
    main()
