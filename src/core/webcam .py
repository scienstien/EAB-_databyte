import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe
mp_selfie = mp.solutions.selfie_segmentation
segment = mp_selfie.SelfieSegmentation(model_selection=1)

# Load background image
bg_image = cv2.imread("C://Users//neera//OneDrive//Desktop//Spider_Laterels_NLP//gaze_env//background.png")
bg_image = cv2.resize(bg_image, (640, 480))

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = segment.process(frame_rgb)
    mask = results.segmentation_mask

    # Create binary mask
    condition = mask > 0.5

    # Resize bg to match frame
    bg = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))

    output = np.where(condition[..., None], frame, bg)

    cv2.imshow("Virtual Background", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
