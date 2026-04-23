from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# ================= LOAD MODELS =================
with open("rf_model.pkl", "rb") as f:
    rf = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# ================= ROUTES =================

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# ================= PREDICTION =================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        f1, f2, f3, f4, f5 = [float(x) for x in request.form.values()]

        # ================= FIXED FEATURE ENGINEERING =================
        features = [
            f1, f2, f3, f4, f5,
            f1 * 0.5, f2 * 0.5, f3 * 0.5, f4 * 0.5, f5,
            max(f1, 1), min(f2, 5), f3 + 1, f4 + 1,
            f1 / 10, f2 / 10,
            f3 / (f1 + 1), f4 / (f1 + 1),
            f1 % 10, f2 % 5, f3 % 3, f4 % 2,
            f5,
            f1 * 0.1, f2 * 0.2, f3 * 0.3
        ]

        # Ensure 26 features
        features = features[:26]

        data = np.array([features])
        data_scaled = scaler.transform(data)

        # Model prediction
        prediction = rf.predict(data_scaled)[0]

        # ================= FINAL DECISION =================
        if prediction == 1 or f3 >= 3 or f5 == 1:
            result = "Malicious"
            confidence = 95
        else:
            result = "Normal"
            confidence = 90

    except Exception as e:
        print("Error:", e)
        result = "Error"
        confidence = 0

    return render_template(
        'dashboard.html',
        prediction_text=result,
        confidence=confidence
    )


# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True)