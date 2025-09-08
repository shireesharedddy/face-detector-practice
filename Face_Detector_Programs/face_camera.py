import cv2
import datetime

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Add overlay text (timestamp + your name)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, f'Shireesha - {timestamp}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # Show the video frame
    cv2.imshow('Camera with Face Detection', frame)

    key = cv2.waitKey(1)

    # Save image when 's' is pressed
    if key == ord('s'):
        filename = f'snapshot_{timestamp.replace(":", "-")}.jpg'
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")

    # Quit on 'q' key
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
