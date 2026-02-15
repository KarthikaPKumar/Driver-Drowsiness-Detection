import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from preprocess import preprocess_frame, extract_features

data = []
labels = []

for label, folder in enumerate(["data/non_targets", "data/targets"]):
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file))
        edges = preprocess_frame(img)
        features = extract_features(edges)
        data.append(features)
        labels.append(label)

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "model/target_model.pkl")
