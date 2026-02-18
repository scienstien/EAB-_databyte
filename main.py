import cv2
import time
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.core.camera import VideoStream
from src.core.analyzer import EmotionAnalyzer
from src.ui.visualizer import Visualizer
from src.utils.fps_counter import FPSCounter

# Try MediaPipe first, fallback to simple detector
try:
    import os
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # Absolute path to your model (avoid Windows path nonsense)
    BASE_DIR = os.path.dirname(__file__)
    MODEL_PATH = os.path.join(BASE_DIR, "models", "blaze_face_short_range.tflite")

    # Create FaceDetector
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    face_detector = vision.FaceDetector.create_from_options(options)

    print("✅ MediaPipe FaceDetector initialized")
    USE_MEDIAPIPE = True


    
except ImportError:
    from src.core.face_detector import SimpleFaceDetector
    USE_MEDIAPIPE = False
    print("⚠️  MediaPipe not available, using OpenCV Haar Cascade fallback")

def main():
    print("🚀 Starting Advanced Emotion Analytics System...")
    
    # 1. Initialize Components
    camera = VideoStream(src=Config.CAMERA_ID, width=Config.CAMERA_WIDTH, height=Config.CAMERA_HEIGHT).start()
    analyzer = EmotionAnalyzer()
    analyzer.start()
    visualizer = Visualizer()
    
    # Face Detection
    if not USE_MEDIAPIPE:
        face_detection = SimpleFaceDetector()

    
    # FPS Calculation
    fps_counter = FPSCounter(window_size=30)
    
    # Recording
    recording = False
    out = None
    
    # Frame counter for throttling
    frame_count = 0
    
    print("✅ System Ready. Press 'q' to exit, 'r' to toggle recording.")
    
    try:
        video_ts = 0
        while True:
            frame = camera.read()
            if frame is None: break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Detect Face
            face_img = None
            face_coords = None
            
            if USE_MEDIAPIPE:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                video_ts += 1
                result = face_detector.detect_for_video(mp_image, video_ts)


                if result.detections is not None and len(result.detections) > 0:
                    for det in result.detections:
                        bbox = det.bounding_box
                        x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

                        x, y = max(0, x), max(0, y)
                        bw, bh = min(w - x, bw), min(h - y, bh)

                        if bw > 0 and bh > 0:
                            face_img = frame[y:y+bh, x:x+bw]
                            face_coords = (x, y, bw, bh)
                            break

            else:
                # OpenCV Haar Cascade detection
                faces = face_detection.detect(frame)
                if len(faces) > 0:
                    x, y, bw, bh = faces[0]  # Use first face
                    face_img = frame[y:y+bh, x:x+bw]
                    face_coords = (x, y, bw, bh)
            
            # 2. Analyze Emotion (throttled for performance)
            if face_img is not None and frame_count % Config.ANALYSIS_THROTTLE == 0:
                analyzer.analyze(face_img)
            
            frame_count += 1
            
            # 3. Get Results
            emotion, probs = analyzer.get_results()
            
            # 4. Visualize
            if face_coords:
                visualizer.draw_face_box(frame, face_coords, emotion)
                
            # Calculate FPS
            fps_counter.update()
            fps = fps_counter.get_fps()
            
            visualizer.draw_hud(frame, emotion, probs, fps)
            
            # Recording Indicator
            if recording:
                cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
                if out: out.write(frame)
            
            cv2.imshow(Config.WINDOW_NAME, frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                recording = not recording
                if recording:
                    filename = f"recording_{int(time.time())}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
                    print(f"🔴 Recording started: {filename}")
                else:
                    if out: out.release()
                    print("⚪ Recording stopped")

    except KeyboardInterrupt:
        pass
    finally:
        print("🛑 Shutting down...")
        camera.stop()
        analyzer.stop()
        if out: out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
