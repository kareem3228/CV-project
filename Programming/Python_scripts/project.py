import cv2
from fer import FER

detector = FER(mtcnn=True)  # use MTCNN for face detection
cap = cv2.VideoCapture(0)

# forimages
# frame= cv2.imread(r"C:\Users\karee\OneDrive\Pictures\Screenshots\Screenshot 2025-05-10 212750.png")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # FER detector returns a list of faces with emotion scores
    results = detector.detect_emotions(frame)


    for face in results:
        x, y, w, h = face["box"]
        emotions = face["emotions"]    # dict of emotion scores
        if emotions["disgust"] > 0.005:
            emotions["disgust"] *= 20
            emotions["fear"] = 0
            emotions["sad"] = 0
            emotions["angry"] = 0
            emotions['happy'] = 0
            emotions["neutral"] = 0
        else:
            emotions["fear"] *= 1.5
            emotions["sad"] *= 2
            emotions["angry"] *= 2.2
            emotions['happy'] *=0.5
            emotions["neutral"] *= 0.5
        # Pick the emotion with highest score
        dominant_emotion = max(emotions, key=emotions.get)

        # Draw rectangle and label on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,0), 2)
        cv2.putText(frame, dominant_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    cv2.imshow("FER Emotion", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#for images
# cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
