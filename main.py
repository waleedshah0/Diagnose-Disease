import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from fuzzywuzzy import process
from tkinter import Tk, Label, Entry, Button, Text, Toplevel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import os


if getattr(sys, 'frozen', False):
    # If the script is running as a bundled executable
    base_path = sys._MEIPASS
else:
    # If running as a normal Python script
    base_path = os.path.dirname(__file__)

disease_model_path = os.path.join(base_path, 'disease_model.h5')
vectorizer_path = os.path.join(base_path, 'vectorizers.pkl')
lable_path = os.path.join(base_path, 'label_encoder.pkl')
csv_path = os.path.join(base_path, 'Disease precaution.csv')
csv_path2 = os.path.join(base_path, 'Symptomsdata.csv')
# Load artifacts
model = load_model(disease_model_path)
with open(vectorizer_path, 'rb') as f:
    vectorizers = pickle.load(f)
with open(lable_path, 'rb') as f:
    label_encoder = pickle.load(f)


precaution_data = pd.read_csv(csv_path)

# Load test data for performance evaluation
test_data = pd.read_csv(csv_path2)  # Assuming the test data is available
symptom_columns = [f'Symptom_{i}' for i in range(1, 18)]  # 17 symptoms

# Preprocess test data
X_test_vectors = []
for column in symptom_columns:
    vectorizer = vectorizers[column]
    X_test_vector = vectorizer.transform(test_data[column].fillna('')).toarray()
    X_test_vectors.append(X_test_vector)
X_test = np.hstack(X_test_vectors)
y_test = label_encoder.transform(test_data['Disease'])



# Evaluate model
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')




metrics_text = (
    f"Model Performance Metrics:\n"
    f"Accuracy: {accuracy:.4f}\n"
    f"Precision: {precision:.4f}\n"
    f"Recall: {recall:.4f}\n"
    f"F1 Score: {f1:.4f}\n"
)

# Preprocessing functions
def preprocess_symptoms(input_text):
    extracted_symptoms = []
    for symptom_column in symptom_columns:
        match, score = process.extractOne(input_text, vectorizers[symptom_column].get_feature_names_out())
        if score >= 75:
            extracted_symptoms.append(match)
    return extracted_symptoms

def vectorize_symptoms(symptoms):
    X_vectors = []
    for vectorizer in vectorizers.values():
        vector = vectorizer.transform([' '.join(symptoms)]).toarray()
        X_vectors.append(vector)
    return np.hstack(X_vectors)

def get_precautions(disease):
    precautions = precaution_data[precaution_data['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten()
    return [p for p in precautions if pd.notnull(p)]

# GUI functions
def predict_disease():
    input_text = entry.get()
    symptoms = preprocess_symptoms(input_text)
    if not symptoms:
        result_label.config(text="No matching symptoms found.")
        return

    input_vector = vectorize_symptoms(symptoms)
    prediction = model.predict(input_vector)[0]  # Get probabilities for all classes

    # Get top 3 diseases with probabilities
    top_3_indices = np.argsort(prediction)[-3:][::-1]
    top_3_diseases = [(label_encoder.inverse_transform([i])[0], prediction[i] * 100) for i in top_3_indices]

    # Prepare result text
    result_text = "Top 3 Predicted Diseases:\n"
    for i, (disease, prob) in enumerate(top_3_diseases, start=1):
        result_text += f"{i}. {disease} - {prob:.2f}%\n"

    # Add precautions for the most probable disease
    most_probable_disease = top_3_diseases[0][0]
    precautions = get_precautions(most_probable_disease)
    if precautions:
        result_text += f"\nPrecautions for {most_probable_disease}:\n"
        result_text += "\n".join(f"- {p}" for p in precautions)
    else:
        result_text += f"\nNo specific precautions available for {most_probable_disease}."

    result_label.config(text=result_text)


def show_metrics():
    metrics_window = Toplevel(root)
    metrics_window.title("Model Performance Metrics")
    Label(metrics_window, text=metrics_text, justify="left").pack(padx=10, pady=10)

# GUI setup
root = Tk()
root.title("Disease Prediction")

Label(root, text="Enter your symptoms:").pack()
entry = Entry(root, width=50)
entry.pack()

Button(root, text="Predict", command=predict_disease).pack()
Button(root, text="Show Model Metrics", command=show_metrics).pack(pady=5)

result_label = Label(root, text="", wraplength=400, justify="left")
result_label.pack()

root.mainloop()
