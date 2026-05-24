"""
realtime_merged_inference.py
Merged single-hand + two-hand realtime inference (uses mediapipe + PyTorch).
Features:
 - Auto-select model based on 1 or 2 hands detected
 - Confidence bar in GUI
 - Green rectangle when prediction becomes stable
 - Inference timing shown in GUI and printed to terminal
 - Sentence builder (appends confirmed labels)
"""

import time
import math
from collections import deque
import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import os
from datetime import datetime

# ===================== CONFIG - edit paths if needed =====================
# By default these point to the same paths used in your uploaded demos.
MODEL_PATH_SINGLE =r"C:\transformer_nikhil_demo\test_new_data_model\trail_demo\new_one_hand_model_8_dim_512_layer_enc_6.pth"   # change if needed
LABEL_MAP_SINGLE_PATH = r"C:\transformer_nikhil_demo\label_map\label_map_signs_one_hand.txt"            # uploaded file. :contentReference[oaicite:6]{index=6}

MODEL_PATH_DUAL = r"C:\transformer_nikhil_demo\test_new_data_model\trail_demo\dual_hand_model_num_head_8_512_LR_enc_6.pth"  # change if needed
LABEL_MAP_DUAL_PATH = r"C:\transformer_nikhil_demo\label_map\label_map_days_twohand.txt"                              # uploaded file. :contentReference[oaicite:7]{index=7}

# If you prefer to store models inside project folder, update MODEL_PATH_* accordingly.

# Settings (kept consistent with your original demos)
TARGET_FPS = 30
FRAME_TIME = 1.0 / TARGET_FPS
MAX_SEQ_LEN = 59
FEATURE_DIM = 126  # both your models use 126 (one-hand: 63 padded, two-hand: 126). See uploaded code. :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

# Per-model thresholds (copied from your demos)
CONF_THRESH_SINGLE = 0.90
STABLE_FRAMES_SINGLE = 6

CONF_THRESH_DUAL = 0.90
STABLE_FRAMES_DUAL = 12

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===================== MODEL CLASSES (copied from your demos) =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SignTransformerEncoder(nn.Module):
    def __init__(self,
                 feature_dim,
                 d_model,
                 nhead,
                 num_layers,
                 ffn_dim,
                 num_classes,
                 dropout,
                 max_seq_len):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc_out(x)

# You can tweak these model architecture hyperparams if your saved weights were trained with different ones.
D_MODEL = 128
N_HEAD = 8
NUM_LAYERS = 6
FFN_DIM = 512
DROPOUT = 0.1

# ===================== LOAD LABEL MAPS =====================
def load_label_map(path):
    idx2label = {}
    if not os.path.exists(path):
        print(f"Warning: label map not found: {path}")
        return idx2label
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                idx = int(parts[0])
                lbl = str(parts[0])
            else:
                idx, lbl = parts
                idx = int(idx)
            idx2label[int(idx)] = lbl
    return idx2label

idx2label_single = load_label_map(LABEL_MAP_SINGLE_PATH)
idx2label_dual   = load_label_map(LABEL_MAP_DUAL_PATH)

print("Single-hand labels loaded:", idx2label_single)
print("Two-hand labels loaded:", idx2label_dual)

# ===================== LOAD MODELS (if model files exist) =====================
def build_and_load(model_path, num_classes):
    model = SignTransformerEncoder(
        FEATURE_DIM, D_MODEL, N_HEAD, NUM_LAYERS, FFN_DIM, num_classes, DROPOUT, MAX_SEQ_LEN
    ).to(device)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded model: {model_path}")
    else:
        print(f"Model file not found at {model_path}. Model architecture created but weights NOT loaded.")
    model.eval()
    return model

model_single = build_and_load(MODEL_PATH_SINGLE, num_classes=len(idx2label_single) or 1)
model_dual   = build_and_load(MODEL_PATH_DUAL,   num_classes=len(idx2label_dual) or 1)

# ===================== MEDIAPIPE =====================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

def extract_onehand_features(results):
    """Use first detected hand only. Pad to FEATURE_DIM."""
    if not results.multi_hand_landmarks:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    hand = results.multi_hand_landmarks[0]
    features = []
    for lm in hand.landmark:
        features.extend([lm.x, lm.y, lm.z])
    features = features[:FEATURE_DIM]
    features += [0.0] * (FEATURE_DIM - len(features))
    return np.array(features, dtype=np.float32)

def extract_twohand_features(results):
    """Use up to two hands, hand0 + hand1 -> 126 features. Pads if needed."""
    feats = []
    if results.multi_hand_landmarks:
        hands = results.multi_hand_landmarks[:2]
        for hand in hands:
            for lm in hand.landmark:
                feats.extend([lm.x, lm.y, lm.z])
    feats = feats[:FEATURE_DIM]
    feats += [0.0] * (FEATURE_DIM - len(feats))
    return np.array(feats, dtype=np.float32)

# ===================== TEMPORAL BUFFERS & STATE =====================
sequence_buffer = []
# Stability buffers separate per model
stability_single = deque(maxlen=STABLE_FRAMES_SINGLE)
stability_dual   = deque(maxlen=STABLE_FRAMES_DUAL)

confirmed_label = None
confirmed_conf  = 0.0
sentence_tokens = []

# To show a transient green rectangle when a label becomes confirmed:
stable_indicator_frames = 0
STABLE_INDICATOR_DURATION = 30  # frames to show green rectangle after confirmation

# ===================== CAMERA LOOP =====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

prev_time = time.time()
print("\n✅ Merged realtime inference started. Press 'q' to quit.\n")

with mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:
    while True:
        # FPS limiter
        elapsed = time.time() - prev_time
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)
        prev_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Draw landmarks (if any)
        if results.multi_hand_landmarks:
            for h in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        # Decide which extractor/model to use
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

        if num_hands >= 2:
            feat = extract_twohand_features(results)
            active_model = "dual"
            conf_thresh = CONF_THRESH_DUAL
            stability_buf = stability_dual
            model = model_dual
            idx2label = idx2label_dual
        elif num_hands == 1:
            feat = extract_onehand_features(results)
            active_model = "single"
            conf_thresh = CONF_THRESH_SINGLE
            stability_buf = stability_single
            model = model_single
            idx2label = idx2label_single
        else:
            feat = np.zeros(FEATURE_DIM, dtype=np.float32)
            active_model = None
            conf_thresh = None
            stability_buf = None
            model = None
            idx2label = {}

        # Append frame features to sequence buffer
        sequence_buffer.append(feat)
        if len(sequence_buffer) > MAX_SEQ_LEN:
            sequence_buffer.pop(0)

        new_label = None
        inference_ms = None

        # Only run inference when we have enough frames
        if model is not None and len(sequence_buffer) == MAX_SEQ_LEN:
            seq = torch.from_numpy(np.stack(sequence_buffer)).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(seq)
                probs = torch.softmax(out, dim=-1)[0].cpu().numpy()
            t1 = time.perf_counter()
            inference_ms = (t1 - t0) * 1000.0

            idx = int(np.argmax(probs))
            conf = float(probs[idx])

            # Stability logic: compare indices in the model's index space
            if conf >= conf_thresh:
                stability_buf.append(idx)
            else:
                stability_buf.clear()

            if len(stability_buf) == stability_buf.maxlen and len(set(stability_buf)) == 1:
                cand_idx = stability_buf[0]
                cand_label = idx2label.get(cand_idx, f"idx_{cand_idx}")
                if cand_label != confirmed_label:
                    confirmed_label = cand_label
                    confirmed_conf = conf
                    new_label = cand_label
                    stable_indicator_frames = STABLE_INDICATOR_DURATION

                    # Print to terminal with timestamp
                    print(f"[{datetime.now().isoformat(timespec='seconds')}] Detected: '{new_label}' "
                          f"model={active_model} conf={conf*100:.2f}% inference_time={inference_ms:.1f}ms")
        # If no model active, clear stability buffers
        elif model is None:
            stability_single.clear()
            stability_dual.clear()

        if new_label:
            sentence_tokens.append(new_label)

        sentence = " ".join(sentence_tokens)

        # ===================== UI DRAWING =====================
        h, w, _ = frame.shape

        # 1) Prediction text and sentence
        cv2.putText(frame,
                    f"Prediction: {confirmed_label or '---'} ({confirmed_conf*100 if confirmed_label else 0:.1f}%)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame,
                    f"Sentence: {sentence or '---'}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

        # 2) FPS (approx)
        fps_display = f"{int(1.0 / max(1e-6, time.time() - prev_time))}"
        cv2.putText(frame,
                    f"FPS: {fps_display}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 3) Inference time (if available)
        if inference_ms is not None:
            cv2.putText(frame,
                        f"Inference: {inference_ms:.1f} ms ({'2H' if active_model=='dual' else '1H'})",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

        # 4) Confidence bar (bottom-left)
        bar_x, bar_y = 10, h - 40
        bar_w, bar_h = 200, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)  # background

        conf_val = confirmed_conf if confirmed_label else 0.0
        fill_w = int(bar_w * conf_val)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), (0,200,0), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (255,255,255), 1)
        cv2.putText(frame, f"Confidence: {conf_val*100:.0f}%", (bar_x+bar_w+10, bar_y+bar_h-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # 5) Green rectangle when prediction recently became stable
        if stable_indicator_frames > 0:
            cv2.rectangle(frame, (w-220, 10), (w-10, 70), (0,200,0), -1)  # filled green
            cv2.putText(frame, "STABLE", (w-200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            stable_indicator_frames -= 1

        # 6) Active model indicator top-right
        model_tag = "2H" if active_model == "dual" else ("1H" if active_model == "single" else "NONE")
        cv2.putText(frame, f"Model: {model_tag}", (w-150, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # Show the window
        cv2.imshow("Merged Sign Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("Session ended.")
