import cv2
import os

VIDEO_PATH = "face_prompt.mp4"  # Replace with your filename if needed

# Check if file exists
if not os.path.exists(VIDEO_PATH):
    print(f"❌ File not found: {VIDEO_PATH}")
    exit()

# Try to open the video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Failed to open the video file.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"📽️ Video opened successfully — {frame_count} frames at {fps:.2f} FPS")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of video.")
        break

    frame_resized = cv2.resize(frame, (640, 360))
    cv2.imshow("🎞️ Test Video Playback", frame_resized)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        print("🚪 Quit by user.")
        break

cap.release()
cv2.destroyAllWindows()
