import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model (4).pkl")
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# Feature categories mappings based on typical encoded patterns
CATEGORICAL_MAPPINGS = {
    "Gender": {"Male": 0, "Female": 1},
    "Blood Type": {"A+": 0, "A-": 1, "B+": 2, "B-": 3, "AB+": 4, "AB-": 5, "O+": 6, "O-": 7},
    "Medical Condition": {"Diabetes": 0, "Asthma": 1, "Obesity": 2, "Arthritis": 3, "Hypertension": 4, "Cancer": 5},
    "Admission Type": {"Emergency": 0, "Elective": 1, "Urgent": 2},
    "Insurance Provider": {"Cigna": 0, "Blue Cross": 1, "Aetna": 2, "UnitedHealthcare": 3, "Medicare": 4},
    "Hospital": {"General Hospital": 0, "City Medical": 1, "St. Jude": 2, "Clinic Center": 3}, 
    "Medication": {"Aspirin": 0, "Ibuprofen": 1, "Penicillin": 2, "Paracetamol": 3, "Lipitor": 4}
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Healthcare Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen flex flex-col justify-between">
    <header class="bg-white border-b border-slate-200 py-5 shadow-sm">
        <div class="max-w-5xl mx-auto px-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="bg-indigo-600 text-white p-2 rounded-lg shadow-md">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                </div>
                <div>
                    <h1 class="text-xl font-bold text-slate-900">HealthPredict AI</h1>
                    <p class="text-xs text-slate-500">Optimized Serverless Diagnostic Engine</p>
                </div>
            </div>
        </div>
    </header>

    <main class="max-w-5xl mx-auto px-4 py-10 w-full flex-grow">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div class="lg:col-span-2 bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
                <h2 class="text-lg font-semibold text-slate-900 mb-5 pb-2 border-b border-slate-100">Patient Intake Profile</h2>
                <form id="predictionForm" class="space-y-6">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Age</label>
                            <input type="number" name="Age" min="0" max="120" value="45" required class="w-full px-3.5 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Billing Amount ($)</label>
                            <input type="number" step="0.01" name="Billing Amount" min="0" value="1500.00" required class="w-full px-3.5 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Gender</label>
                            <select name="Gender" class="w-full px-3.5 py-2 border border-slate-300 rounded-lg bg-white">
                                {% for item in mappings['Gender'].keys() %} <option value="{{ item }}">{{ item }}</option> {% endfor %}
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Blood Type</label>
                            <select name="Blood Type" class="w-full px-3.5 py-2 border border-slate-300 rounded-lg bg-white">
                                {% for item in mappings['Blood Type'].keys() %} <option value="{{ item }}">{{ item }}</option> {% endfor %}
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Medical Condition</label>
                            <select name="Medical Condition" class="w-full px-3.5 py-2 border border-slate-300 rounded-lg bg-white">
                                {% for item in mappings['Medical Condition'].keys() %} <option value="{{ item }}">{{ item }}</option> {% endfor %}
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Admission Type</label>
                            <select name="Admission Type" class="w-full px-3.5 py-2 border border-slate-300 rounded-lg bg-white">
                                {% for item in mappings['Admission Type'].keys() %} <option value="{{ item }}">{{ item }}</option> {% endfor %}
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Insurance Provider</label>
                            <select name="Insurance Provider" class="w-full px-3.5 py-2 border border-slate-300 rounded-lg bg-white">
                                {% for item in mappings['Insurance Provider'].keys() %} <option value="{{ item }}">{{ item }}</option> {% endfor %}
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-slate-700 mb-1">Medication</label>
                            <select name="Medication" class="w-full px-3.5 py-2 border border-slate-300 rounded-lg bg-white">
                                {% for item in mappings['Medication'].keys() %} <option value="{{ item }}">{{ item }}</option> {% endfor %}
                            </select>
                        </div>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-1">Hospital Facility</label>
                        <select name="Hospital" class="w-full px-3.5 py-2 border border-slate-300 rounded-lg bg-white">
                            {% for item in mappings['Hospital'].keys() %} <option value="{{ item }}">{{ item }}</option> {% endfor %}
                        </select>
                    </div>
                    <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2.5 px-4 rounded-lg shadow-md cursor-pointer text-center">
                        Evaluate Analytics Instance
                    </button>
                </form>
            </div>

            <div class="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm flex flex-col justify-between">
                <div>
                    <h2 class="text-lg font-semibold text-slate-900 mb-5 pb-2 border-b border-slate-100">Inference Diagnostics</h2>
                    <div id="placeholder" class="text-center py-20 text-slate-400">
                        <p class="text-sm">Submit profile metrics to populate data diagnostics.</p>
                    </div>
                    <div id="resultsCard" class="hidden space-y-6">
                        <div class="bg-indigo-50 border border-indigo-100 rounded-xl p-5 text-center">
                            <span class="text-xs font-semibold uppercase tracking-wider text-indigo-600 block mb-1">Prediction Verdict</span>
                            <div id="predictionValue" class="text-3xl font-extrabold text-indigo-900">--</div>
                        </div>
                        <div id="probabilitiesContainer" class="space-y-3"></div>
                    </div>
                </div>
            </div>
        </div>
    </main>
    <footer class="bg-white border-t border-slate-200 py-4 text-center text-xs text-slate-500">&copy; 2026 HealthPredict AI.</footer>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const rawData = Object.fromEntries(formData.entries());
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(rawData)
                });
                const data = await response.json();
                if (data.success) {
                    document.getElementById('placeholder').classList.add('hidden');
                    document.getElementById('resultsCard').classList.remove('hidden');
                    document.getElementById('predictionValue').innerText = `Class ${data.prediction}`;
                    
                    const container = document.getElementById('probabilitiesContainer');
                    container.innerHTML = '';
                    data.probabilities.forEach((prob, idx) => {
                        const pct = (prob * 100).toFixed(1);
                        container.innerHTML += `
                            <div>
                                <div class="flex justify-between text-xs font-medium text-slate-600 mb-1"><span>Class ${idx}</span><span>${pct}%</span></div>
                                <div class="w-full bg-slate-100 rounded-full h-2"><div class="bg-indigo-600 h-2 rounded-full" style="width: ${pct}%"></div></div>
                            </div>`;
                    });
                }
            } catch (err) { alert("Error connecting to engine."); }
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, mappings=CATEGORICAL_MAPPINGS)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model file missing."}), 500
    try:
        data = request.get_json()
        
        # Enforce explicit exact structural order matching the 9 XGBoost features
        features = [
            int(data["Age"]),
            CATEGORICAL_MAPPINGS["Gender"][data["Gender"]],
            CATEGORICAL_MAPPINGS["Blood Type"][data["Blood Type"]],
            CATEGORICAL_MAPPINGS["Medical Condition"][data["Medical Condition"]],
            CATEGORICAL_MAPPINGS["Hospital"][data["Hospital"]],
            CATEGORICAL_MAPPINGS["Insurance Provider"][data["Insurance Provider"]],
            float(data["Billing Amount"]),
            CATEGORICAL_MAPPINGS["Admission Type"][data["Admission Type"]],
            CATEGORICAL_MAPPINGS["Medication"][data["Medication"]]
        ]
        
        # Format explicitly as a 2D numpy array row matching Scikit/XGBoost structures
        input_array = np.array([features], dtype=np.float32)
        
        prediction = int(model.predict(input_array)[0])
        probabilities = model.predict_proba(input_array)[0].tolist()
        
        return jsonify({"success": True, "prediction": prediction, "probabilities": probabilities})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400
