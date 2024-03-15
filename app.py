import cv2

# Load the cascade
#cascade_classifier = cv2.CascadeClassifier(r'C:\Users\meena\OneDrive\Desktop\Hackathon\haarcascade_frontalface_default.xml')
cascade_classifier = cv2.CascadeClassifier(r'C:\Users\meena\OneDrive\Desktop\Hackathon\haarcascade_eye.xml')

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    detections = cascade_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the faces
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()