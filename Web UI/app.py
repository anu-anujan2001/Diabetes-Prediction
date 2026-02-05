from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model once
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

FIELDS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    # Default empty values (so GET request won't fail)
    form_data = {k: "" for k in FIELDS}

    if request.method == "POST":
        # Get form data safely
        for k in FIELDS:
            form_data[k] = request.form.get(k, "")

        # Convert to correct numeric types for model
        sample = pd.DataFrame([{
            "Pregnancies": int(form_data["Pregnancies"]),
            "Glucose": float(form_data["Glucose"]),
            "BloodPressure": float(form_data["BloodPressure"]),
            "SkinThickness": float(form_data["SkinThickness"]),
            "Insulin": float(form_data["Insulin"]),
            "BMI": float(form_data["BMI"]),
            "DiabetesPedigreeFunction": float(form_data["DiabetesPedigreeFunction"]),
            "Age": int(form_data["Age"])
        }])

        pred = model.predict(sample)[0]
        prob = model.predict_proba(sample)[0][1]  # probability of class 1

        prediction = "Diabetic" if pred == 1 else "Non-Diabetic"
        probability = round(prob * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        form_data=form_data
    )

if __name__ == "__main__":
    app.run(debug=True)
