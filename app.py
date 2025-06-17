from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
from flask import make_response
from xhtml2pdf import pisa
from io import BytesIO
from datetime import datetime
from flask import jsonify

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session management

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




@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    name = request.form['name']
    gender = request.form['gender']
    age = request.form['age']
    height = request.form['height']
    weight = request.form['weight']
    bmi = request.form['bmi']
    bmi_class = request.form['bmi_class']
    tip = request.form['tip']

    # Auto-generate today's date
    date_today = datetime.today().strftime("%d-%m-%Y")

    # Render the HTML with all values
    html = render_template('pdf.html',
                           name=name,
                           gender=gender,
                           age=age,
                           height=height,
                           weight=weight,
                           bmi=bmi,
                           bmi_class=bmi_class,
                           tip=tip,
                           date=date_today)

    # Generate PDF
    result = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=result)

    # Return PDF file as response
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

        # Store result in session and redirect
        session['result'] = {
            'name': name,
            'bmi': round(bmi, 2),
            'bmi_class': bmi_class,
            'tip': tip
        }
        return redirect(url_for('result'))

    return render_template('index.html')


@app.route('/result')
def result():
    result = session.get('result')
    if not result:
        return redirect(url_for('home'))
    return render_template('result.html', **result)


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
