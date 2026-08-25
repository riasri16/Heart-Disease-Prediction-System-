# ============================================================
# app.py — Flask Backend with Report Generation
# ============================================================

from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import json
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

app = Flask(__name__)

# Load Models & Scalers
model1  = joblib.load('model1_ecg_clinical.pkl')
scaler1 = joblib.load('scaler1_ecg_clinical.pkl')
model2  = joblib.load('model2_clinical_only.pkl')
scaler2 = joblib.load('scaler2_clinical_only.pkl')

# -------------------------------------------------------
# HELPER — Run Prediction
# -------------------------------------------------------
def run_prediction(form):
    age         = int(form['age'])
    sex         = int(form['sex'])
    chest_pain  = int(form['chest_pain'])
    resting_bp  = int(form['resting_bp'])
    cholesterol = int(form['cholesterol'])
    fasting_bs  = int(form['fasting_bs'])
    resting_ecg = int(form['resting_ecg'])
    max_hr      = int(form['max_hr'])
    ex_angina   = int(form['ex_angina'])
    oldpeak     = float(form['oldpeak'])
    st_slope    = int(form['st_slope'])
    height      = int(form['height'])
    weight      = float(form['weight'])
    ap_lo       = int(form['ap_lo'])
    gluc        = int(form['gluc'])
    smoke       = int(form['smoke'])
    alco        = int(form['alco'])
    active      = int(form['active'])

    if cholesterol < 200:   chol_m2 = 1
    elif cholesterol < 240: chol_m2 = 2
    else:                   chol_m2 = 3

    gender_m2 = 1 if sex == 1 else 2

    sample1        = np.array([[age, sex, chest_pain, resting_bp,
                                cholesterol, fasting_bs, resting_ecg,
                                max_hr, ex_angina, oldpeak, st_slope]])
    sample2        = np.array([[age, gender_m2, height, weight,
                                resting_bp, ap_lo, chol_m2,
                                gluc, smoke, alco, active]])

    sample1_scaled = scaler1.transform(sample1)
    pred1          = model1.predict(sample1_scaled)[0]
    prob1          = model1.predict_proba(sample1_scaled)[0]

    sample2_scaled = scaler2.transform(sample2)
    pred2          = model2.predict(sample2_scaled)[0]
    prob2          = model2.predict_proba(sample2_scaled)[0]

    if pred1 == 1 and pred2 == 1:
        verdict       = "HIGH RISK"
        verdict_color = "red"
    elif pred1 == 0 and pred2 == 0:
        verdict       = "LOW RISK"
        verdict_color = "green"
    else:
        verdict       = "MODERATE RISK CONSULT A DOCTOR"
        verdict_color = "orange"

    # Recommendations based on verdict
    if verdict_color == "red":
        recommendations = [
            "🏥 Consult a cardiologist immediately",
            "💊 Review current medications with your doctor",
            "🚫 Avoid strenuous physical activity until evaluated",
            "🥗 Follow a strict low-sodium, low-fat diet",
            "🚭 Stop smoking and avoid alcohol completely",
            "📊 Monitor blood pressure and heart rate daily",
            "😴 Get at least 7-8 hours of sleep per night",
            "🧘 Reduce stress through meditation or yoga"
        ]
    elif verdict_color == "orange":
        recommendations = [
            "👨‍⚕️ Schedule a check-up with your doctor soon",
            "🥗 Adopt a heart-healthy diet (fruits, vegetables, whole grains)",
            "🏃 Start light exercise like walking 30 mins/day",
            "🚭 Reduce or eliminate smoking and alcohol",
            "⚖️ Maintain a healthy weight (BMI 18.5-24.9)",
            "📊 Monitor cholesterol and blood pressure regularly",
            "💧 Stay well hydrated — drink 8 glasses of water daily",
            "😴 Improve sleep quality and reduce stress"
        ]
    else:
        recommendations = [
            "✅ Keep maintaining your healthy lifestyle",
            "🏃 Continue regular physical activity (150 mins/week)",
            "🥗 Maintain a balanced and nutritious diet",
            "📊 Get annual heart check-ups as a precaution",
            "🚭 Stay smoke-free and limit alcohol intake",
            "⚖️ Maintain healthy weight and BMI",
            "💧 Stay hydrated and manage stress levels",
            "😴 Ensure 7-8 hours of quality sleep nightly"
        ]

    bmi = round(weight / ((height / 100) ** 2), 1)

    return {
        'age'           : age,
        'sex'           : 'Male' if sex == 1 else 'Female',
        'chest_pain'    : ['ATA','NAP','ASY','TA'][chest_pain],
        'resting_bp'    : resting_bp,
        'cholesterol'   : cholesterol,
        'fasting_bs'    : 'Yes' if fasting_bs == 1 else 'No',
        'resting_ecg'   : ['Normal','ST Abnormality','LVH'][resting_ecg],
        'max_hr'        : max_hr,
        'ex_angina'     : 'Yes' if ex_angina == 1 else 'No',
        'oldpeak'       : oldpeak,
        'st_slope'      : ['Up','Flat','Down'][st_slope],
        'height'        : height,
        'weight'        : weight,
        'ap_lo'         : ap_lo,
        'gluc'          : ['','Normal','Above Normal','Well Above Normal'][gluc],
        'smoke'         : 'Yes' if smoke == 1 else 'No',
        'alco'          : 'Yes' if alco == 1 else 'No',
        'active'        : 'Yes' if active == 1 else 'No',
        'bmi'           : bmi,
        'pred1'         : int(pred1),
        'prob1_no'      : round(prob1[0] * 100, 2),
        'prob1_dis'     : round(prob1[1] * 100, 2),
        'pred2'         : int(pred2),
        'prob2_no'      : round(prob2[0] * 100, 2),
        'prob2_dis'     : round(prob2[1] * 100, 2),
        'verdict'       : verdict,
        'verdict_color' : verdict_color,
        'recommendations': recommendations,
        'date'          : datetime.now().strftime("%d %B %Y, %I:%M %p")
    }

# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        try:
            data = run_prediction(request.form)
            return jsonify({
                'success'       : True,
                'model1_pred'   : data['pred1'],
                'model1_no_dis' : data['prob1_no'],
                'model1_dis'    : data['prob1_dis'],
                'model2_pred'   : data['pred2'],
                'model2_no_dis' : data['prob2_no'],
                'model2_dis'    : data['prob2_dis'],
                'verdict'       : data['verdict'],
                'verdict_color' : data['verdict_color']
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

@app.route('/report', methods=['POST'])
def report():
    try:
        data = run_prediction(request.form)
        return render_template('report.html', data=data)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        data = run_prediction(request.form)

        buffer = io.BytesIO()
        doc    = SimpleDocTemplate(buffer, pagesize=A4,
                                   rightMargin=40, leftMargin=40,
                                   topMargin=40, bottomMargin=40)

        styles  = getSampleStyleSheet()
        content = []

        # Colors
        BLUE   = colors.HexColor('#1a73e8')
        RED    = colors.HexColor('#d93025')
        GREEN  = colors.HexColor('#188038')
        ORANGE = colors.HexColor('#e37400')
        GRAY   = colors.HexColor('#666666')
        LIGHT  = colors.HexColor('#f8f9fa')

        verdict_color_map = {
            'red'   : RED,
            'green' : GREEN,
            'orange': ORANGE
        }
        vc = verdict_color_map[data['verdict_color']]

        # Title Style
        title_style = ParagraphStyle('title',
            fontSize=24, fontName='Helvetica-Bold',
            textColor=BLUE, alignment=TA_CENTER, spaceAfter=5)

        sub_style = ParagraphStyle('sub',
            fontSize=11, fontName='Helvetica',
            textColor=GRAY, alignment=TA_CENTER, spaceAfter=20)

        section_style = ParagraphStyle('section',
            fontSize=13, fontName='Helvetica-Bold',
            textColor=BLUE, spaceAfter=10, spaceBefore=15)

        normal_style = ParagraphStyle('normal',
            fontSize=10, fontName='Helvetica',
            textColor=colors.black, spaceAfter=5)

        verdict_style = ParagraphStyle('verdict',
            fontSize=22, fontName='Helvetica-Bold',
            textColor=vc, alignment=TA_CENTER, spaceAfter=5)

        rec_style = ParagraphStyle('rec',
            fontSize=10, fontName='Helvetica',
            textColor=colors.black, spaceAfter=4, leftIndent=10)

        # ── HEADER ──
        content.append(Paragraph("🫀 IntelliHeart", title_style))
        content.append(Paragraph("Cardiovascular Disease Prediction Report", sub_style))
        content.append(Paragraph(f"Generated: {data['date']}", sub_style))
        content.append(HRFlowable(width="100%", thickness=1, color=BLUE))
        content.append(Spacer(1, 15))

        # ── PATIENT INFO ──
        content.append(Paragraph("Patient Information", section_style))

        patient_data = [
            ['Parameter', 'Value', 'Parameter', 'Value'],
            ['Age',            str(data['age']) + ' years',  'Sex',          data['sex']],
            ['Height',         str(data['height']) + ' cm',  'Weight',       str(data['weight']) + ' kg'],
            ['BMI',            str(data['bmi']),              'Smoker',       data['smoke']],
            ['Alcohol',        data['alco'],                  'Active',       data['active']],
            ['Resting BP',     str(data['resting_bp']) + ' mmHg', 'Diastolic BP', str(data['ap_lo']) + ' mmHg'],
            ['Cholesterol',    str(data['cholesterol']) + ' mg/dl', 'Glucose',  data['gluc']],
            ['Fasting BS>120', data['fasting_bs'],            'Chest Pain',  data['chest_pain']],
        ]

        pt = Table(patient_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        pt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), BLUE),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 10),
            ('BACKGROUND', (0,1), (-1,-1), LIGHT),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT, colors.white]),
            ('GRID',       (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('FONTNAME',   (0,1), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME',   (2,1), (2,-1), 'Helvetica-Bold'),
            ('PADDING',    (0,0), (-1,-1), 7),
        ]))
        content.append(pt)
        content.append(Spacer(1, 15))

        # ── ECG PARAMETERS ──
        content.append(Paragraph("ECG Parameters", section_style))

        ecg_data = [
            ['ECG Parameter',    'Value'],
            ['Resting ECG',      data['resting_ecg']],
            ['Max Heart Rate',   str(data['max_hr']) + ' bpm'],
            ['Exercise Angina',  data['ex_angina']],
            ['ST Depression',    str(data['oldpeak'])],
            ['ST Slope',         data['st_slope']],
        ]

        et = Table(ecg_data, colWidths=[3*inch, 3*inch])
        et.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1557b0')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 10),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT, colors.white]),
            ('GRID',       (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('FONTNAME',   (0,1), (0,-1), 'Helvetica-Bold'),
            ('PADDING',    (0,0), (-1,-1), 7),
        ]))
        content.append(et)
        content.append(Spacer(1, 15))

        # ── MODEL RESULTS ──
        content.append(Paragraph("AI Model Results", section_style))

        m1_color = RED if data['pred1'] == 1 else GREEN
        m2_color = RED if data['pred2'] == 1 else GREEN
        m1_text  = "HEART DISEASE DETECTED" if data['pred1'] == 1 else "NO HEART DISEASE"
        m2_text  = "DISEASE DETECTED" if data['pred2'] == 1 else "NO DISEASE"

        results_data = [
            ['Model',                    'Result',   'No Disease %', 'Heart Disease %'],
            ['Model 1 (ECG + Clinical)', m1_text,   str(data['prob1_no']) + '%', str(data['prob1_dis']) + '%'],
            ['Model 2 (Clinical Only)',  m2_text,   str(data['prob2_no']) + '%', str(data['prob2_dis']) + '%'],
        ]

        rt = Table(results_data, colWidths=[2.2*inch, 2*inch, 1.4*inch, 1.4*inch])
        rt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), BLUE),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 10),
            ('TEXTCOLOR',  (1,1), (1,1), m1_color),
            ('TEXTCOLOR',  (1,2), (1,2), m2_color),
            ('FONTNAME',   (1,1), (1,-1), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [LIGHT, colors.white]),
            ('GRID',       (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('PADDING',    (0,0), (-1,-1), 7),
        ]))
        content.append(rt)
        content.append(Spacer(1, 15))

        # ── VERDICT ──
        content.append(HRFlowable(width="100%", thickness=1, color=vc))
        content.append(Spacer(1, 10))
        content.append(Paragraph("COMBINED VERDICT", sub_style))
        content.append(Paragraph(data['verdict'], verdict_style))
        content.append(Spacer(1, 10))
        content.append(HRFlowable(width="100%", thickness=1, color=vc))
        content.append(Spacer(1, 15))

        # ── RECOMMENDATIONS ──
        content.append(Paragraph("Recommendations", section_style))
        for rec in data['recommendations']:
            content.append(Paragraph(rec, rec_style))

        content.append(Spacer(1, 20))
        content.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
        content.append(Spacer(1, 8))
        content.append(Paragraph(
            "⚠️ This report is generated by an AI prediction system and is for educational purposes only. "
            "Always consult a qualified medical professional for diagnosis and treatment.",
            ParagraphStyle('disclaimer', fontSize=8, textColor=GRAY, alignment=TA_CENTER)
        ))

        doc.build(content)
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'IntelliHeart_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )

    except Exception as e:
        return f"PDF Error: {str(e)}"

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)