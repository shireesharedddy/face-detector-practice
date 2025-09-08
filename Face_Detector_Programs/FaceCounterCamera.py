import cv2

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Count faces
    face_count = len(faces)

    # Decide rectangle color based on count
    if face_count == 1:
        rect_color = (0, 255, 0)     # Green
    elif 2 <= face_count <= 3:
        rect_color = (0, 165, 255)   # Orange
    elif face_count >= 4:
        rect_color = (0, 0, 255)     # Red
    else:
        rect_color = (255, 255, 255) # White (no faces)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)

    # Display face count on screen
    cv2.putText(frame, f"Faces: {face_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, rect_color, 2)

    # Show the frame
    cv2.imshow('Face Counter Camera', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
