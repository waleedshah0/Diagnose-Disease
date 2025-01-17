# Diagnose-Disease
Diagnose Disease



# NLP Disease Diagnosis

This project is an NLP-based disease diagnosis application that predicts potential diseases based on user-input symptoms and provides corresponding precautionary measures.

## Features
- Predicts the top 3 most probable diseases based on input symptoms.
- Displays the probabilities for each predicted disease.
- Provides precautionary measures for the most probable disease.
- Interactive GUI built with Tkinter.

## Requirements
Ensure you have Python installed (>=3.7). Install the required libraries using the `requirements.txt` file.

### Installation Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure all data files (`Symptomsdata.csv`, `Disease precaution.csv`) and saved artifacts (`disease_model.h5`, `vectorizers.pkl`, `label_encoder.pkl`) are present in the project directory.

4. Run the application:
   ```bash
   python main.py
   ```

## Files

### `symptoms.py`
- Prepares and trains the disease diagnosis model using a dataset of symptoms and corresponding diseases.
- Vectorizes symptoms using `TfidfVectorizer`.
- Encodes disease labels and trains a neural network model.

### `main.py`
- Implements the GUI for the application.
- Loads the trained model and artifacts.
- Accepts user input for symptoms and predicts diseases using the trained model.
- Displays the top 3 predicted diseases along with precautions for the most probable disease.

## Dataset
1. **Symptoms Data:**
   - `Symptomsdata.csv` contains symptom information mapped to diseases.
2. **Precaution Data:**
   - `Disease precaution.csv` contains precautionary measures for various diseases.

## Evaluation Metrics
- **Accuracy**: Measures the correct predictions of the model.
- **Loss**: Represents the model's error during training.

## Challenges
- Handling multiple symptoms in a single input string.
- Ensuring high accuracy for diseases with sparse data.
- Mapping user-input symptoms to dataset symptoms using fuzzy matching.

## Future Considerations
- Improve the model by integrating advanced NLP techniques such as transformers or embeddings.
- Add support for more symptoms and diseases.
- Deploy the application as a web service for better accessibility.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

# Contributing
Feel free to open issues or contribute via pull requests.

