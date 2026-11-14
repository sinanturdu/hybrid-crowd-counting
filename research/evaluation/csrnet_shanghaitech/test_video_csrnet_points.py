import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from models.csrnet import CSRNet


# ======================================================
# CONFIG
# ======================================================
VIDEO_PATH = os.getenv("VIDEO_PATH", "examples/sample_video.mp4")
WEIGHTS_PATH = os.path.join("weights", "csrnet_AB_best.pth")

SAVE_OUTPUT = True
OUTPUT_PATH = "csrnet_video_output.mp4"

PATCH_SIZE = 384
STRIDE = 192

# Local maxima (nokta) ayarları
POINT_MIN_DIST = 6
POINT_THR_REL = 0.10

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ======================================================
# LOCAL MAXIMA (POINT EXTRACTION)
# ======================================================
def extract_points_from_density(density, min_dist=6, thr_rel=0.10):
    d = density.copy()
    d[d < 0] = 0

    mx = float(d.max())
    if mx <= 0:
        return []

    thr = mx * thr_rel
    points = []

    while True:
        idx = int(np.argmax(d))
        y, x = np.unravel_index(idx, d.shape)

        if float(d[y, x]) < thr:
            break

        points.append((int(y), int(x)))

        y0 = max(0, y - min_dist)
        y1 = min(d.shape[0], y + min_dist + 1)
        x0 = max(0, x - min_dist)
        x1 = min(d.shape[1], x + min_dist + 1)
        d[y0:y1, x0:x1] = 0.0

    return points


# ======================================================
# GAUSSIAN WEIGHT (PATCH BLENDING)
# ======================================================
_WEIGHT_CACHE = {}

def gaussian_weight(h, w, sigma=0.5):
    key = (h, w)
    if key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[key]

    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    d = np.sqrt(xx**2 + yy**2)
    g = np.exp(-(d**2) / (2 * sigma**2))
    wt = torch.from_numpy(g.astype(np.float32))
    _WEIGHT_CACHE[key] = wt
    return wt


# ======================================================
# PATCH-BASED INFERENCE (FRAME)
# ======================================================
@torch.no_grad()
def infer_frame(model, frame_rgb, device):
    model.eval()

    img_h, img_w = frame_rgb.shape[:2]
    img_pil = Image.fromarray(frame_rgb)
    img_tensor = to_tensor(img_pil).to(device)

    out_h = int(math.ceil(img_h / 8.0))
    out_w = int(math.ceil(img_w / 8.0))

    density_map = torch.zeros((out_h, out_w), device=device)
    norm_map = torch.zeros((out_h, out_w), device=device)

    tops = list(range(0, max(1, img_h - PATCH_SIZE + 1), STRIDE))
    lefts = list(range(0, max(1, img_w - PATCH_SIZE + 1), STRIDE))

    if tops[-1] != max(0, img_h - PATCH_SIZE):
        tops.append(max(0, img_h - PATCH_SIZE))
    if lefts[-1] != max(0, img_w - PATCH_SIZE):
        lefts.append(max(0, img_w - PATCH_SIZE))

    for top in tops:
        for left in lefts:
            patch = img_tensor[:, top:top + PATCH_SIZE, left:left + PATCH_SIZE]

            ph, pw = patch.shape[1], patch.shape[2]
            if ph < PATCH_SIZE or pw < PATCH_SIZE:
                patch = F.pad(patch, (0, PATCH_SIZE - pw, 0, PATCH_SIZE - ph))

            pred = model(patch.unsqueeze(0))
            pred = torch.clamp(F.relu(pred), max=5.0)
            pred = pred.squeeze(0).squeeze(0)

            real_h = min(PATCH_SIZE, img_h - top)
            real_w = min(PATCH_SIZE, img_w - left)
            real_oh = int(math.ceil(real_h / 8.0))
            real_ow = int(math.ceil(real_w / 8.0))

            pred = pred[:real_oh, :real_ow]

            y0, x0 = top // 8, left // 8
            weight = gaussian_weight(*pred.shape).to(device)

            y1 = min(out_h, y0 + pred.shape[0])
            x1 = min(out_w, x0 + pred.shape[1])

            density_map[y0:y1, x0:x1] += pred[:y1-y0, :x1-x0] * weight[:y1-y0, :x1-x0]
            norm_map[y0:y1, x0:x1] += weight[:y1-y0, :x1-x0]

    density_map = density_map / (norm_map + 1e-6)

    pred_low = density_map.cpu().numpy()
    pred_count = float(pred_low.sum())

    return pred_low, pred_count


# ======================================================
# MAIN
# ======================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Device:", device)

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(VIDEO_PATH)

    model = CSRNet(load_pretrained_vgg=False).to(device)
    ckpt = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pred_low, pred_count = infer_frame(model, frame_rgb, device)
        points = extract_points_from_density(
            pred_low,
            min_dist=POINT_MIN_DIST,
            thr_rel=POINT_THR_REL
        )

        # low-res -> full-res ölçek
        scale_y = H / pred_low.shape[0]
        scale_x = W / pred_low.shape[1]

        for (y, x) in points:
            cx = int(x * scale_x)
            cy = int(y * scale_y)
            cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"Count: {pred_count:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            2
        )

        if SAVE_OUTPUT:
            writer.write(frame)

        cv2.imshow("CSRNet Crowd Counting", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx} | Count: {pred_count:.1f}")

    cap.release()
    if SAVE_OUTPUT:
        writer.release()
    cv2.destroyAllWindows()

    print("🎉 Video inference finished")


if __name__ == "__main__":
    main()
