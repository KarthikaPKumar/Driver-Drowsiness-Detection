import cv2
import joblib
import numpy as np
from preprocess import preprocess_frame, extract_features

model = joblib.load("model/target_model.pkl")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    edges = preprocess_frame(frame)
    features = extract_features(edges).reshape(1, -1)
    prediction = model.predict(features)
    confidence = model.predict_proba(features)[0][1]

    label = "TARGET" if prediction == 1 else "NO TARGET"
    color = (0,255,0) if prediction == 1 else (0,0,255)

    cv2.putText(frame, f"{label} ({confidence:.2f})", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Real-Time AI Target Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
