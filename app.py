import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.datasets._lfw import fetch_lfw_people
from sklearn.utils import Bunch
from collections import deque

lfw_people: Bunch = fetch_lfw_people(min_faces_per_person=70) # type: ignore

# Loading model
model = load_model('models/model_2.keras')

face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

target = lfw_people.target_names

history_size = 10
preds_history = deque(maxlen=history_size)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    THRESHOLD = 0.8

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (37, 50)) 
        roi_normalized = roi_resized / 255.0
        input_data = np.reshape(roi_normalized, (1, 50, 37, 1))

        prediction = model.predict(input_data, verbose=0)

        preds_history.append(prediction[0])

        # smoothed_prediction = np.mean(preds_history, axis=0)

        result_idx = np.argmax(prediction)
        confidence = np.max(prediction)

        name = target[result_idx]
        if confidence > THRESHOLD:
            label = f"{name}: {confidence*100:.1f}%"
            color = (0, 0, 255)
        else:
            label = f"other: {confidence*100:.1f}%"
            color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Face Recognition System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()