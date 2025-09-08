import cv2

# Load pre-trained Haar Cascade face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read video frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale (needed for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the frame in a window
    cv2.imshow('Face Detection - Press Q to Quit', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
