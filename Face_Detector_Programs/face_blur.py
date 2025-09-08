import cv2
from datetime import datetime
import time

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
last_snapshot_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Get current timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Overlay live timestamp
    cv2.putText(frame, timestamp, (10, 30), font, 0.6, (255, 255, 255), 1)

    # Blur each face
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(face_roi, (99, 99), 30)
        frame[y:y+h, x:x+w] = blurred
        # Optional: draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show face count on screen
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), font, 0.6, (0, 255, 255), 2)

    # Take snapshot if 3+ faces and 5 seconds passed
    if len(faces) >= 3 and (time.time() - last_snapshot_time) > 5:
        snapshot_filename = f"snapshot_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(snapshot_filename, frame)
        print(f"📸 Snapshot saved as: {snapshot_filename}")
        last_snapshot_time = time.time()

    # Display frame
    cv2.imshow("Face Camera with Blur", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
