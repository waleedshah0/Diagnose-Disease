import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load datasets
symptom_data = pd.read_csv('Symptomsdata.csv')
precaution_data = pd.read_csv('Disease precaution.csv')

# Preprocessing
symptom_columns = [f'Symptom_{i}' for i in range(1, 18)]  # 17 symptoms
disease_labels = symptom_data['Disease']

# Vectorize symptoms
vectorizers = {}
X_vectors = []

for column in symptom_columns:
    vectorizer = TfidfVectorizer(max_features=100)
    X_vector = vectorizer.fit_transform(symptom_data[column].fillna('')).toarray()
    vectorizers[column] = vectorizer
    X_vectors.append(X_vector)

X = np.hstack(X_vectors)

# Encode disease labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(disease_labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = Sequential([
    Dense(128, input_dim=X.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nModel Performance Metrics:")
print(f"Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save artifacts
model.save('disease_model.h5')
with open('vectorizers.pkl', 'wb') as f:
    pickle.dump(vectorizers, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Training complete. Model and artifacts saved.")
