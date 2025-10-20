import cv2
import mediapipe as mp

# Initialize MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Try to open default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Check /dev/video0 permissions or camera index.")

print("Camera opened successfully. Press 'q' to exit.")

# Create a MediaPipe Hands detector
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    iter =0
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        # Flip horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the image color to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Hands
        results = hands.process(rgb)

        # Draw results on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                print(f'found {iter}')
                iter=iter +1

        # Display the output
        cv2.imshow("MediaPipe Test", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Exited MediaPipe test.")