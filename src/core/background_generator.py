import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

class BackgroundGenerator:
    def __init__(self):
        self.CONF_THRESHOLD = 0.7
        self.emotion_history = deque(maxlen=15)
        self.current_background = "neutral"

        BASE_DIR = os.path.dirname(__file__)
        BG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..","..","backgrounds"))
        BASE_DIR = os.path.dirname(__file__)
        MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models", "selfie_segmenter.tflite"))

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Segmentation model not found: {MODEL_PATH}")

        # Load backgrounds
        self.backgrounds = {
            "happy": cv2.imread(os.path.join(BG_DIR, "happy.jpeg")),
            "sad": cv2.imread(os.path.join(BG_DIR, "sad.jpeg")),
            "angry": cv2.imread(os.path.join(BG_DIR, "angry.jpeg")),
            "neutral": cv2.imread(os.path.join(BG_DIR, "neutral.jpeg")),
            "fear" :  cv2.imread(os.path.join(BG_DIR, "fear.jpeg")),
            "surprise" :  cv2.imread(os.path.join(BG_DIR, "surprise.jpeg"))
        }

        for k, v in self.backgrounds.items():
            if v is None:
                print(f"⚠️ Background image for '{k}' failed to load.")

        # Initialize Image Segmenter (Tasks API)
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

        self.video_ts = 0

    def apply(self, frame, probs):
        if probs is None or not probs:
            return frame

        # Stability logic for background switching
        top_emotion = max(probs, key=probs.get)
        if probs[top_emotion] > self.CONF_THRESHOLD:
            self.emotion_history.append(top_emotion)

        if len(self.emotion_history) == self.emotion_history.maxlen:
            stable_emotion = max(set(self.emotion_history), key=self.emotion_history.count)
            self.current_background = stable_emotion

        bg_image = self.backgrounds.get(self.current_background)
        if bg_image is None:
            return frame

        # Run segmentation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self.video_ts += 1
        result = self.segmenter.segment_for_video(mp_image, self.video_ts)

        mask = result.category_mask
        if mask is None:
            return frame

        mask_np = mask.numpy_view()

        # 🔥 Normalize mask shape to (H, W)
        mask_np = np.squeeze(mask_np)

        if mask_np.ndim != 2:
            raise ValueError(f"Unexpected mask shape after squeeze: {mask_np.shape}")

        # Binary mask
        condition = mask_np > 0.65

        # Expand to (H, W, 1) for RGB broadcasting
        condition = condition[..., None]

        bg_resized = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))

        output = np.where(condition, bg_resized, frame)

        return output

