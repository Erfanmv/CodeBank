import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import gradio as gr

# Load and preprocess the data
df = pd.read_csv('hospital.csv')
X = df.drop('output', axis=1)
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Classifier': SVC(probability=True)
}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} - Classification Report:")
    print(classification_report(y_test, y_pred))

# Comparison of Model Performance
results = pd.DataFrame({
    'Model': models.keys(),
    'Accuracy': [accuracy_score(y_test, model.predict(X_test)) for model in models.values()],
    'Precision': [precision_score(y_test, model.predict(X_test)) for model in models.values()],
    'Recall': [recall_score(y_test, model.predict(X_test)) for model in models.values()],
    'F1-Score': [f1_score(y_test, model.predict(X_test)) for model in models.values()]
})
print("\nModel Evaluation Metrics:")
print(results.to_markdown(index=False, numalign="left", stralign="left"))

# Function to interpret probability
def interpret_probability(probability):
    if probability >= 0.8:
        return "Very High Risk"
    elif 0.6 <= probability < 0.8:
        return "High Risk"
    elif 0.4 <= probability < 0.6:
        return "Moderate Risk"
    elif 0.2 <= probability < 0.4:
        return "Low Risk"
    else:
        return "Very Low Risk"

# Function for prediction with Gradio interface
def predict_heart_attack_gradio(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall):
    sex_mapping = {"Male": 1, "Female": 0}
    cp_mapping = {"0-Typical Angina": 0, "1-Atypical Angina": 1, "2-Non-anginal Pain": 2, "3-Asymptomatic": 3}
    fbs_mapping = {"True": 1, "False": 0}
    exng_mapping = {"Yes": 1, "No": 0}
    slp_mapping = {"0-Upsloping": 0, "1-Flat": 1, "2-Downsloping": 2}
    thall_mapping = {"1-Normal": 1, "2-Fixed Defect": 2, "3-Reversable Defect": 3}

    sex = sex_mapping[sex]
    cp = cp_mapping[cp]
    fbs = fbs_mapping[fbs]
    exng = exng_mapping[exng]
    slp = slp_mapping[slp]
    thall = thall_mapping[thall]

    input_values_scaled = scaler.transform([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])
    
    results = {}
    probabilities = []
    for name, model in models.items():
        probability = model.predict_proba(input_values_scaled)[0][1]
        probabilities.append(probability)
        interpretation = interpret_probability(probability)
        results[name] = (probability, interpretation)
        
    overall_probability = np.mean(probabilities)
    overall_interpretation = interpret_probability(overall_probability)
    
    result_str = ""
    for model_name, (prob, interp) in results.items():
        result_str += f"<h3>{model_name}</h3><p>Probability of Heart Attack: {prob:.2f} ({interp})</p>"
    
    result_str += f"<h2>Overall Probability of Heart Attack</h2><p>{overall_probability:.2f} ({overall_interpretation})</p>"
    
    return result_str

# Gradio Interface with Enhanced Features
theme = gr.themes.Monochrome()
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("## Heart Attack Prediction")
    
    # Description from the provided image
    gr.Markdown("""
    This tool predicts the likelihood of a heart attack based on several medical inputs using three different models: Logistic Regression, Random Forest, and Support Vector Classifier.
    Each model provides a probability of heart attack, which is interpreted into risk categories such as 'Very High Risk', 'High Risk', etc. An overall probability is also calculated by averaging the results from all models.
    """)
    
    # Inputs in a grid layout
    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Age of the patient", value=54)
            sex = gr.Dropdown(label="Sex (1: Male, 0: Female)", choices=["Male", "Female"], value="Male")
            trtbps = gr.Number(label="Resting Blood Pressure (in mm Hg on admission to the hospital)\n< 120: Normal\n120-129: Elevated\n130-139: Hypertension Stage 1\n>= 140: Hypertension Stage 2", value=130)
            chol = gr.Number(label="Serum Cholesterol (in mg/dl)\n< 200: Desirable\n200-239: Borderline high\n>= 240: High", value=240)
            oldpeak = gr.Number(label="ST Depression Induced by Exercise Relative to Rest (mm)\n>= 2: Significant ST depression, may indicate ischemia", value=0.8)

        with gr.Column():
            cp = gr.Radio(label="Chest Pain Type (0-3)\n0: Typical angina (chest pain related to reduced blood flow to the heart)\n1: Atypical angina (chest pain not related to heart)\n2: Non-anginal pain (not related to heart)\n3: Asymptomatic (no chest pain)", choices=["0-Typical Angina", "1-Atypical Angina", "2-Non-anginal Pain", "3-Asymptomatic"], value="0-Typical Angina")
            fbs = gr.Radio(label="Fasting Blood Sugar > 120 mg/dl\n0: False (Normal)\n1: True (High)", choices=["True", "False"], value="False")
            restecg = gr.Radio(label="Resting ECG Results\n0: Normal\n1: Having ST-T wave abnormality\n2: Showing probable or definite left ventricular hypertrophy",choices=["0", "1", "2"], value="1")
            thalachh = gr.Number(label="Maximum Heart Rate Achieved (beats per minute)\nVaries: Depends on age and fitness level. Lower values may indicate heart problems.", value=153)
            exng = gr.Radio(label="Exercise Induced Angina\n0: No\n1: Yes", choices=["Yes", "No"], value="No")
        
        with gr.Column():
            slp = gr.Radio(label="Slope of the Peak Exercise ST Segment (0-2)\n0: Upsloping (generally normal)\n1: Flat (may indicate ischemia)\n2: Downsloping (often associated with severe ischemia)", choices=["0-Upsloping", "1-Flat", "2-Downsloping"], value="0-Upsloping")
            caa = gr.Radio(label="Number of Major Vessels Colored by Fluoroscopy (0-3)\n0: None\n1-3: Increasing number of vessels with significant blockage", choices=["0", "1", "2", "3"], value="0")
            thall = gr.Radio(label="Thalassemia (1-3)\n1: Normal\n2: Fixed defect (prior heart damage)\n3: Reversible defect (may indicate ischemia)", choices=["1-Normal", "2-Fixed Defect", "3-Reversable Defect"], value="1-Normal")
          

    predict_button = gr.Button("Predict")
    output = gr.HTML(label="Prediction Results")

    predict_button.click(predict_heart_attack_gradio, inputs=[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall], outputs=output)

demo.launch(share=True)
