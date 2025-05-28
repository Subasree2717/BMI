from flask import Flask, render_template, request
import joblib
import numpy as np
from flask import make_response
from xhtml2pdf import pisa
from io import BytesIO

app = Flask(__name__)

bmi_results = []

# Load models
model = joblib.load('bmi_classifier.pkl')
scaler = joblib.load('bmi_scaler.pkl')
label_encoder = joblib.load('bmi_label_encoder.pkl')

bmi_tips = {
    "Normal weight": "Great job! Maintain a healthy lifestyle.",
    "Overweight": "Consider regular exercise and a balanced diet.",
    "Obese Class 1": "Start a supervised fitness and diet plan.",
    "Obese Class 2": "Consult a healthcare provider for assistance.",
    "Obese Class 3": "High health risk — professional help is strongly advised.",
    "Underweight": "Eat more nutritious meals and consult a dietitian."
}


@app.route('/bmi_chart')
def bmi_chart():
    from collections import Counter
    classes = [rec['bmi_class'] for rec in bmi_results]
    counter = dict(Counter(classes))
    return {"labels": list(counter.keys()), "values": list(counter.values())}



@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    name = request.form['name']
    bmi = request.form['bmi']
    bmi_class = request.form['bmi_class']

    html = render_template('pdf.html', name=name, bmi=bmi, bmi_class=bmi_class)
    result = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=result)

    response = make_response(result.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=bmi_report.pdf'
    return response

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        gender = request.form['gender']
        age = int(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])

        bmi = weight / ((height / 100) ** 2)
        X_scaled = scaler.transform([[age, height, weight]])
        prediction = model.predict(X_scaled)
        bmi_class = label_encoder.inverse_transform(prediction)[0]
        tip = bmi_tips.get(bmi_class, "Stay healthy!")
        
        bmi_results.append({'name': name, 'bmi_class': bmi_class})

        return render_template('index.html', bmi=round(bmi, 2), bmi_class=bmi_class, name=name, tip=tip)

        return render_template('index.html', bmi=round(bmi, 2), bmi_class=bmi_class, name=name, tip=tip)

    return render_template('index.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

