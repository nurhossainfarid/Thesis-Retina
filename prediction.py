import os, json
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

MODEL_NAME  = "hybridmodel" 
MODEL_PATH  = r"Result/FLResult/HybridModel/last_global_model.pth"
IMG_PATH    = r"predicted_outputs/Lung-CTScan_Copy/Malignant cases/Malignant_cases_22.jpg" 

OUT_DIR     = "./xai_out"
CLASS_NAMES = ['Cataract', 'Diabetic retinopathy', 'Glaucoma', 'Healthy']
IMG_SIZE    = 224
NORM_MEAN, NORM_STD = 0.5, 0.5          
ALPHA = 0.35                          
METHOD = "smoothgrad_campp"              
SMOOTHGRAD_SAMPLES = 10                  
SMOOTHGRAD_NOISE_STD = 0.10            

from models.model_factory import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# robust checkpoint loader
def load_checkpoint_flex(model: nn.Module, path: str, device: torch.device):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict):
        for k in ["model_state_dict", "state_dict", "weights", "model"]:
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
        # keep only tensor-like if mixed
        if not all(isinstance(v, torch.Tensor) for v in sd.values()):
            sd = {k:v for k,v in sd.items() if isinstance(v, torch.Tensor)} or sd
    new_sd = {}
    for k,v in sd.items():
        if k.startswith("module."): k = k[7:]
        if k.startswith("model."):  k = k[6:]
        new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:   print("Missing:", missing)
    if unexpected:print("Unexpected:", unexpected)

# build & load model ----
model = get_model(MODEL_NAME, num_classes=len(CLASS_NAMES), pretrained=False).to(device).eval()
load_checkpoint_flex(model, MODEL_PATH, device)

# image io ----
def load_ct_gray(path: str, out_size: Tuple[int,int]=(IMG_SIZE, IMG_SIZE)):
    img_u8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise ValueError(f"Could not load: {path}")
    orig = img_u8.copy()
    img = cv2.resize(img_u8, out_size, interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    img = (img - NORM_MEAN) / (NORM_STD if NORM_STD!=0 else 1.0)
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return orig, x

# utils ----
def normalize01(a: np.ndarray):
    a = a.astype(np.float32)
    a -= a.min()
    a += 1e-12
    a /= a.max()
    return a

def overlay_on_gray(img_gray_u8: np.ndarray, heat: np.ndarray, alpha: float = 0.35):
    H,W = img_gray_u8.shape
    heat_r = cv2.resize(heat, (W, H))
    heatmap = cv2.applyColorMap((heat_r*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    base = cv2.cvtColor(img_gray_u8, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(base, 1.0, heatmap, alpha, 0)

# target layer auto-pick (last Conv2d)
def find_last_conv(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d found for CAM.")
    return last

target_layer = find_last_conv(model)

# Guided Backprop (for guided Grad-CAM)
class GuidedReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x)

class GuidedBackprop:
    def __init__(self, model: nn.Module):
        self.model = model
        self.relu_handles = []
        self._register()

    def _register(self):
        def relu_backward_hook(module, grad_in, grad_out):
            # only pass positive gradients/activations
            return (torch.clamp(grad_in[0], min=0.0),)
        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                self.relu_handles.append(m.register_full_backward_hook(relu_backward_hook))

    def generate(self, x: torch.Tensor, class_idx: int):
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        score = logits[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)
        gb = x.grad.detach().squeeze(0).squeeze(0).cpu().numpy()  # [H,W]
        gb = np.abs(gb)
        gb = normalize01(gb)
        return gb

    def close(self):
        for h in self.relu_handles:
            h.remove()

# CAM engines
class CAMBase:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.ha = target_layer.register_forward_hook(self._save_act)
        self.hg = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out 
    def _save_grad(self, module, gin, gout):
        self.gradients = gout[0] 

    def close(self):
        self.ha.remove(); self.hg.remove()

    def _post(self, cam: torch.Tensor):
        cam = torch.relu(cam).detach().cpu().numpy()
        return normalize01(cam)

class GradCAMEngine(CAMBase):
    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1))
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        A = self.activations[0]             
        dYdA = self.gradients[0]            
        weights = dYdA.mean(dim=(1,2))     
        cam = (weights.view(-1,1,1) * A).sum(dim=0)
        return self._post(cam)

class GradCAMPlusPlusEngine(CAMBase):
    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1))
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        A = self.activations[0]            
        dY = self.gradients[0]            
        dY2 = dY ** 2
        dY3 = dY2 * dY

        eps = 1e-6
        sum_A = (A * dY3).sum(dim=(1,2))   
        denom = 2.0 * dY2 + sum_A[:, None, None]
        denom = torch.where(denom != 0, denom, torch.ones_like(denom)*eps)
        alpha = dY2 / denom                 
        weights = (alpha * torch.relu(dY)).sum(dim=(1,2))  
        cam = (weights.view(-1,1,1) * A).sum(dim=0)       
        return self._post(cam)

def smoothgrad_campp(model, target_layer, x: torch.Tensor, class_idx: int, samples=10, noise_std=0.1):
    engine = GradCAMPlusPlusEngine(model, target_layer)
    cams = []
    with torch.no_grad():
        base = x.clone().detach()
    for _ in range(samples):
        noise = torch.randn_like(base) * noise_std
        cam = engine.generate(base + noise, class_idx=class_idx)
        cams.append(cam)
    engine.close()
    cams = np.stack(cams, axis=0)
    cam = np.mean(cams, axis=0)
    return normalize01(cam)

# Predict + Explain wrapper
def predict_and_explain(
    model: nn.Module,
    img_path: str,
    target_layer: nn.Module,
    method: str = "smoothgrad_campp",
    alpha: float = 0.35,
    smooth_samples: int = 10,
    smooth_noise: float = 0.1,
) -> Dict:
    os.makedirs(OUT_DIR, exist_ok=True)

    orig_u8, x = load_ct_gray(img_path)
    x = x.to(device)

    # predict
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]

    # explain
    if method == "gradcam":
        engine = GradCAMEngine(model, target_layer)
        heat = engine.generate(x, class_idx=pred_idx)
        engine.close()
    elif method == "gradcampp":
        engine = GradCAMPlusPlusEngine(model, target_layer)
        heat = engine.generate(x, class_idx=pred_idx)
        engine.close()
    elif method == "smoothgrad_campp":
        heat = smoothgrad_campp(model, target_layer, x, pred_idx, samples=smooth_samples, noise_std=smooth_noise)
    elif method == "guided_gradcam":
        # Grad-CAM++ * GuidedBackprop fusion (Hadamard)
        engine = GradCAMPlusPlusEngine(model, target_layer)
        heat_cam = engine.generate(x, class_idx=pred_idx)
        engine.close()

        gb = GuidedBackprop(model)
        gb_map = gb.generate(x, class_idx=pred_idx) 
        gb.close()

        heat = normalize01(heat_cam * gb_map)
    else:
        raise ValueError("Unknown method")

    overlay = overlay_on_gray(orig_u8, heat, alpha=alpha)

    # save
    stem = os.path.splitext(os.path.basename(img_path))[0]
    np.save(os.path.join(OUT_DIR, f"{stem}_{method}_heat.npy"), heat)
    cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_{method}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    with open(os.path.join(OUT_DIR, f"{stem}_probs.json"), "w") as f:
        json.dump({CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}, f, indent=2)

    # show
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(orig_u8, cmap='gray'); plt.title("CT Slice"); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(overlay); plt.title(f"{method.upper()} → {pred_name}"); plt.axis('off')
    plt.show()

    return {
        "pred_idx": pred_idx,
        "pred_name": pred_name,
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "heat": heat,
        "overlay_path": os.path.join(OUT_DIR, f"{stem}_{method}_overlay.png"),
    }

# RUN ON ONE IMAGE
res = predict_and_explain(
    model, IMG_PATH, target_layer,
    method=METHOD,
    alpha=ALPHA,
    smooth_samples=SMOOTHGRAD_SAMPLES,
    smooth_noise=SMOOTHGRAD_NOISE_STD
)
print("Done:", json.dumps({"pred": res["pred_name"], "overlay": res["overlay_path"]}, indent=2))
