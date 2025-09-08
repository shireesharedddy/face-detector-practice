import cv2
from datetime import datetime

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam
cap = cv2.VideoCapture(0)

# To avoid taking multiple snapshots for same frame
snapshot_taken = False

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces (returns list of rectangles)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw bounding boxes for each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Add a live clock (timestamp) overlay
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, f"Time: {timestamp}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Take snapshot if 3 or more faces are detected and not already taken
    if len(faces) >= 3 and not snapshot_taken:
        snapshot_filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(snapshot_filename, frame)
        print(f"Snapshot saved: {snapshot_filename}")
        snapshot_taken = True
    elif len(faces) < 3:
        snapshot_taken = False  # Reset snapshot flag

    # Display the frame with everything
    cv2.imshow('Face Camera with Clock and Snapshot', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and destroy window
cap.release()
cv2.destroyAllWindows()
