import cv2
import numpy as np
import os
import torch

# =========================================================
# Gradient-based CAM (input-gradient)
# =========================================================
def gradcam_simple(model, tensor):
    tensor = tensor.clone().detach().requires_grad_(True)
    model.zero_grad()

    output = model(tensor)
    output.backward()

    grads = tensor.grad.detach()[0]   # (4, H, W)
    cam = grads.abs().mean(dim=0).cpu().numpy()

    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    return cam


# =========================================================
# CAM energy inside / outside lungs
# =========================================================
def cam_lung_stats(cam, lung_mask):
    cam_res = cv2.resize(cam, (lung_mask.shape[1], lung_mask.shape[0]))
    lung_mask = lung_mask.astype(bool)

    total = cam_res.sum() + 1e-8
    inside = cam_res[lung_mask].sum() / total
    outside = cam_res[~lung_mask].sum() / total

    return inside, outside


# =========================================================
# Save XAI visual outputs
#   - Lung mask overlay (cool blue / cyan)
#   - Grad-CAM heatmap
#   - CAM overlay
# =========================================================
def save_xai_images(orig_gray, cam, lung_mask, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)

    h, w = orig_gray.shape

    # ---------- Original image ----------
    orig_norm = cv2.normalize(orig_gray, None, 0, 255, cv2.NORM_MINMAX)
    orig_bgr = cv2.cvtColor(orig_norm, cv2.COLOR_GRAY2BGR)

    # ---------- Grad-CAM heatmap ----------
    cam_res = cv2.resize(cam, (w, h))
    cam_uint8 = (cam_res * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # ---------- CAM overlay ----------
    cam_overlay = cv2.addWeighted(orig_bgr, 0.65, heatmap, 0.35, 0)

    # =====================================================
    # Lung mask overlay (cool blue / cyan, same as reference)
    # =====================================================
    lung_mask = (lung_mask > 0).astype(np.uint8)

    # Cool blue / cyan colour (BGR)
    overlay_color = np.zeros((h, w, 3), dtype=np.uint8)
    overlay_color[:, :, 0] = 180  # Blue
    overlay_color[:, :, 1] = 200  # Green (cyan tone)

    alpha = 0.30  # transparency

    mask_overlay = orig_bgr.copy()
    for c in range(3):
        mask_overlay[:, :, c] = np.where(
            lung_mask == 1,
            (1 - alpha) * orig_bgr[:, :, c] + alpha * overlay_color[:, :, c],
            orig_bgr[:, :, c]
        )

    mask_overlay = mask_overlay.astype(np.uint8)

    # ---------- Save images ----------
    paths = {
        "heatmap": f"{prefix}_cam.png",
        "overlay": f"{prefix}_overlay.png",
        "mask": f"{prefix}_lungmask.png"
    }

    cv2.imwrite(os.path.join(save_dir, paths["heatmap"]), heatmap)
    cv2.imwrite(os.path.join(save_dir, paths["overlay"]), cam_overlay)
    cv2.imwrite(os.path.join(save_dir, paths["mask"]), mask_overlay)

    return paths
