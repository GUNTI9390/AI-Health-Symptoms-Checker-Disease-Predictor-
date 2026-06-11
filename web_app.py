from flask import Flask, render_template_string, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import numpy as np
from joblib import load
import json
import os
import uuid
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import requests as http_requests

app = Flask(__name__)
app.secret_key = os.urandom(24)

HF_API_KEY = "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"
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
USERS_FILE = os.path.join(DATA_DIR, 'users.json')
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model', 'decision_tree.joblib')
os.makedirs(DATA_DIR, exist_ok=True)

def load_users():
    if not os.path.exists(USERS_FILE):
        default = {
            'admin': {
                'id': 'admin', 'username': 'admin', 'email': 'admin@healthapp.com',
                'password': generate_password_hash('admin123'),
                'role': 'admin', 'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'predictions': []
            }
        }
        with open(USERS_FILE, 'w') as f: json.dump(default, f, indent=2)
        return default
    with open(USERS_FILE, 'r') as f: return json.load(f)

def save_users(data):
    with open(USERS_FILE, 'w') as f: json.dump(data, f, indent=2)

model = load(MODEL_PATH)

SYMPTOMS = [
    'itching',
            'skin_rash',
            'nodal_skin_eruptions',
            'continuous_sneezing',
            'shivering',
            'chills',
            'joint_pain',
            'stomach_pain',
            'acidity',
            'ulcers_on_tongue',
            'muscle_wasting',
            'vomiting',
            'burning_micturition',
            'spotting_ urination',
            'fatigue',
            'weight_gain',
            'anxiety',
            'cold_hands_and_feets',
            'mood_swings',
            'weight_loss',
            'restlessness',
            'lethargy',
            'patches_in_throat',
            'irregular_sugar_level',
            'cough',
            'high_fever',
            'sunken_eyes',
            'breathlessness',
            'sweating',
            'dehydration',
            'indigestion',
            'headache',
            'yellowish_skin',
            'dark_urine',
            'nausea',
            'loss_of_appetite',
            'pain_behind_the_eyes',
            'back_pain',
            'constipation',
            'abdominal_pain',
            'diarrhoea',
            'mild_fever',
            'yellow_urine',
            'yellowing_of_eyes',
            'acute_liver_failure',
            'fluid_overload',
            'swelling_of_stomach',
            'swelled_lymph_nodes',
            'malaise',
            'blurred_and_distorted_vision',
            'phlegm',
            'throat_irritation',
            'redness_of_eyes',
            'sinus_pressure',
            'runny_nose',
            'congestion',
            'chest_pain',
            'weakness_in_limbs',
            'fast_heart_rate',
            'pain_during_bowel_movements',
            'pain_in_anal_region',
            'bloody_stool',
            'irritation_in_anus',
            'neck_pain',
            'dizziness',
            'cramps',
            'bruising',
            'obesity',
            'swollen_legs',
            'swollen_blood_vessels',
            'puffy_face_and_eyes',
            'enlarged_thyroid',
            'brittle_nails',
            'swollen_extremeties',
            'excessive_hunger',
            'extra_marital_contacts',
            'drying_and_tingling_lips',
            'slurred_speech',
            'knee_pain',
            'hip_joint_pain',
            'muscle_weakness',
            'stiff_neck',
            'swelling_joints',
            'movement_stiffness',
            'spinning_movements',
            'loss_of_balance',
            'unsteadiness',
            'weakness_of_one_body_side',
            'loss_of_smell',
            'bladder_discomfort',
            'foul_smell_of urine',
            'continuous_feel_of_urine',
            'passage_of_gases',
            'internal_itching',
            'toxic_look_(typhos)',
            'depression',
            'irritability',
            'muscle_pain',
            'altered_sensorium',
            'red_spots_over_body',
            'belly_pain',
            'abnormal_menstruation',
            'dischromic _patches',
            'watering_from_eyes',
            'increased_appetite',
            'polyuria',
            'family_history',
            'mucoid_sputum',
            'rusty_sputum',
            'lack_of_concentration',
            'visual_disturbances',
            'receiving_blood_transfusion',
            'receiving_unsterile_injections',
            'coma',
            'stomach_bleeding',
            'distention_of_abdomen',
            'history_of_alcohol_consumption',
            'fluid_overload.1',
            'blood_in_sputum',
            'prominent_veins_on_calf',
            'palpitations',
            'painful_walking',
            'pus_filled_pimples',
            'blackheads',
            'scurring',
            'skin_peeling',
            'silver_like_dusting',
            'small_dents_in_nails',
            'inflammatory_nails',
            'blister',
            'red_sore_around_nose',
            'yellow_crust_ooze'
]

DISEASE_DISPLAY_MAP = {
  "Fungal infection": "Fungal Infection",
  "Allergy": "Allergy",
  "GERD": "Gerd",
  "Chronic cholestasis": "Chronic Cholestasis",
  "Drug Reaction": "Drug Reaction",
  "Peptic ulcer disease": "Peptic Ulcer Disease",
  "AIDS": "Aids",
  "Diabetes ": "Diabetes",
  "Gastroenteritis": "Gastroenteritis",
  "Bronchial Asthma": "Bronchial Asthma",
  "Hypertension ": "Hypertension",
  "Migraine": "Migraine",
  "Cervical spondylosis": "Cervical Spondylosis",
  "Paralysis (brain hemorrhage)": "Paralysis (Brain Hemorrhage)",
  "Jaundice": "Jaundice",
  "Malaria": "Malaria",
  "Chicken pox": "Chicken Pox",
  "Dengue": "Dengue",
  "Typhoid": "Typhoid",
  "hepatitis A": "Hepatitis A",
  "Hepatitis B": "Hepatitis B",
  "Hepatitis C": "Hepatitis C",
  "Hepatitis D": "Hepatitis D",
  "Hepatitis E": "Hepatitis E",
  "Alcoholic hepatitis": "Alcoholic Hepatitis",
  "Tuberculosis": "Tuberculosis",
  "Common Cold": "Common Cold",
  "Pneumonia": "Pneumonia",
  "Dimorphic hemmorhoids(piles)": "Dimorphic Hemorrhoids (Piles)",
  "Heart attack": "Heart Attack",
  "Varicose veins": "Varicose Veins",
  "Hypothyroidism": "Hypothyroidism",
  "Hyperthyroidism": "Hyperthyroidism",
  "Hypoglycemia": "Hypoglycemia",
  "Osteoarthristis": "Osteoarthritis",
  "Arthritis": "Arthritis",
  "(vertigo) Paroymsal  Positional Vertigo": "Paroxysmal Positional Vertigo",
  "Acne": "Acne",
  "Urinary tract infection": "Urinary Tract Infection",
  "Psoriasis": "Psoriasis",
  "Impetigo": "Impetigo"
}

DISEASE_INFO = {
  "Fungal infection": {
    "desc": "Common skin infection caused by fungi.",
    "prec": [
      "Keep skin dry and clean",
      "Avoid sharing towels or clothes",
      "Wear loose cotton clothing",
      "Use antifungal cream as prescribed"
    ],
    "doc": "Dermatologist",
    "sev": "Low"
  },
  "Allergy": {
    "desc": "Immune system reaction to substances like food, pollen, or medication.",
    "prec": [
      "Identify and avoid your triggers",
      "Keep windows closed during high pollen",
      "Use air purifier at home",
      "Carry antihistamines if prescribed"
    ],
    "doc": "Allergist",
    "sev": "Low"
  },
  "GERD": {
    "desc": "Chronic acid reflux where stomach acid flows back into the esophagus.",
    "prec": [
      "Avoid large meals and eating late",
      "Do not lie down right after meals",
      "Reduce spicy and fatty foods",
      "Elevate the head of your bed"
    ],
    "doc": "Gastroenterologist",
    "sev": "Medium"
  },
  "Chronic cholestasis": {
    "desc": "A condition where bile flow from the liver is reduced for an extended period.",
    "prec": [
      "Avoid alcohol completely",
      "Eat low-fat, nutritious meals",
      "Stay well hydrated",
      "Get regular liver function tests"
    ],
    "doc": "Hepatologist",
    "sev": "High"
  },
  "Drug Reaction": {
    "desc": "An adverse reaction to a medication you have taken.",
    "prec": [
      "Stop the suspected medication",
      "Inform your prescribing doctor",
      "Note when the reaction started",
      "Keep a record of all medications"
    ],
    "doc": "General Physician",
    "sev": "Medium"
  },
  "Peptic ulcer disease": {
    "desc": "Sores that develop on the lining of the stomach or small intestine.",
    "prec": [
      "Avoid NSAIDs like ibuprofen",
      "Eat smaller, frequent meals",
      "Manage stress levels",
      "Avoid smoking and alcohol"
    ],
    "doc": "Gastroenterologist",
    "sev": "Medium"
  },
  "AIDS": {
    "desc": "Acquired immunodeficiency syndrome caused by HIV, which weakens the immune system.",
    "prec": [
      "Follow antiretroviral therapy strictly",
      "Maintain personal hygiene",
      "Get regular blood tests",
      "Avoid exposure to infections"
    ],
    "doc": "Infectious Disease Specialist",
    "sev": "High"
  },
  "Diabetes ": {
    "desc": "A metabolic condition where blood sugar levels are elevated due to insulin issues.",
    "prec": [
      "Monitor your blood sugar regularly",
      "Follow a diabetic diet",
      "Exercise at least 30 minutes daily",
      "Take medication on schedule"
    ],
    "doc": "Endocrinologist",
    "sev": "High"
  },
  "Gastroenteritis": {
    "desc": "Inflammation of the stomach and intestines, usually from a virus or bacteria.",
    "prec": [
      "Stay hydrated with ORS solutions",
      "Avoid dairy products temporarily",
      "Eat bland foods like rice and toast",
      "Wash hands frequently"
    ],
    "doc": "General Physician",
    "sev": "Low"
  },
  "Bronchial Asthma": {
    "desc": "A chronic condition causing airway inflammation, narrowing, and excess mucus.",
    "prec": [
      "Use your inhaler as prescribed",
      "Avoid smoke and dust triggers",
      "Get the yearly flu vaccine",
      "Monitor your peak flow readings"
    ],
    "doc": "Pulmonologist",
    "sev": "Medium"
  },
  "Hypertension ": {
    "desc": "High blood pressure that increases the risk of heart disease and stroke.",
    "prec": [
      "Reduce sodium in your diet",
      "Exercise 30 minutes daily",
      "Limit alcohol to moderate levels",
      "Monitor your blood pressure regularly"
    ],
    "doc": "Cardiologist",
    "sev": "High"
  },
  "Migraine": {
    "desc": "Severe recurring headaches often accompanied by nausea and light sensitivity.",
    "prec": [
      "Identify and avoid your triggers",
      "Stay hydrated throughout the day",
      "Maintain a regular sleep schedule",
      "Practice stress management"
    ],
    "doc": "Neurologist",
    "sev": "Medium"
  },
  "Cervical spondylosis": {
    "desc": "Age-related wear and tear affecting the cervical spine disks.",
    "prec": [
      "Maintain good posture while working",
      "Do gentle neck exercises daily",
      "Avoid prolonged screen time",
      "Use a supportive pillow"
    ],
    "doc": "Orthopedic",
    "sev": "Medium"
  },
  "Paralysis (brain hemorrhage)": {
    "desc": "Loss of muscle function caused by bleeding within brain tissue.",
    "prec": [
      "Seek emergency medical care",
      "Attend regular neurological checkups",
      "Follow a physiotherapy program",
      "Keep blood pressure controlled"
    ],
    "doc": "Neurologist",
    "sev": "Critical"
  },
  "Jaundice": {
    "desc": "Yellowing of the skin and eyes due to elevated bilirubin levels.",
    "prec": [
      "Get plenty of rest",
      "Stay well hydrated",
      "Avoid alcohol completely",
      "Eat light, nutritious meals"
    ],
    "doc": "Hepatologist",
    "sev": "Medium"
  },
  "Malaria": {
    "desc": "A mosquito-borne infectious disease causing high fever, chills, and sweating.",
    "prec": [
      "Sleep under mosquito nets",
      "Apply insect repellent",
      "Complete the full medication course",
      "Drain standing water around your home"
    ],
    "doc": "General Physician",
    "sev": "Medium"
  },
  "Chicken pox": {
    "desc": "A highly contagious viral infection that causes an itchy blister-like rash.",
    "prec": [
      "Do not scratch the blisters",
      "Apply calamine lotion for relief",
      "Keep fingernails trimmed short",
      "Stay home until scabs form"
    ],
    "doc": "General Physician",
    "sev": "Low"
  },
  "Dengue": {
    "desc": "A mosquito-borne viral infection causing high fever and severe joint pain.",
    "prec": [
      "Rest and drink plenty of fluids",
      "Do not take aspirin",
      "Use paracetamol for fever relief",
      "Monitor your platelet count"
    ],
    "doc": "General Physician",
    "sev": "High"
  },
  "Typhoid": {
    "desc": "A bacterial infection spread through contaminated food or water.",
    "prec": [
      "Complete the full antibiotic course",
      "Eat only thoroughly cooked food",
      "Drink boiled or purified water",
      "Get the typhoid vaccine"
    ],
    "doc": "General Physician",
    "sev": "Medium"
  },
  "hepatitis A": {
    "desc": "A liver infection caused by the hepatitis A virus, spread through contaminated food.",
    "prec": [
      "Get plenty of rest",
      "Avoid alcohol completely",
      "Eat small, frequent meals",
      "Practice thorough hand hygiene"
    ],
    "doc": "Hepatologist",
    "sev": "Medium"
  },
  "Hepatitis B": {
    "desc": "A liver infection caused by the hepatitis B virus, spread through blood and fluids.",
    "prec": [
      "Do not share personal items like razors",
      "Do not donate blood or organs",
      "Get regular liver monitoring",
      "Have family members tested"
    ],
    "doc": "Hepatologist",
    "sev": "High"
  },
  "Hepatitis C": {
    "desc": "A viral infection that causes liver inflammation, often becoming chronic.",
    "prec": [
      "Avoid alcohol completely",
      "Take prescribed antiviral medications",
      "Get regular blood work done",
      "Do not share needles or razors"
    ],
    "doc": "Hepatologist",
    "sev": "High"
  },
  "Hepatitis D": {
    "desc": "A serious liver disease that only occurs in people who also have hepatitis B.",
    "prec": [
      "Get vaccinated for hepatitis B",
      "Avoid all alcohol",
      "Attend regular medical followups",
      "Monitor liver function tests"
    ],
    "doc": "Hepatologist",
    "sev": "High"
  },
  "Hepatitis E": {
    "desc": "A liver infection typically spread through contaminated drinking water.",
    "prec": [
      "Drink only safe, clean water",
      "Wash hands before eating",
      "Avoid raw or undercooked shellfish",
      "Get adequate rest"
    ],
    "doc": "Hepatologist",
    "sev": "Medium"
  },
  "Alcoholic hepatitis": {
    "desc": "Liver inflammation caused by heavy and prolonged alcohol consumption.",
    "prec": [
      "Stop drinking alcohol immediately",
      "Get nutritional support",
      "Attend regular liver checkups",
      "Consider a counseling program"
    ],
    "doc": "Hepatologist",
    "sev": "High"
  },
  "Tuberculosis": {
    "desc": "A bacterial infection that primarily affects the lungs and spreads through air.",
    "prec": [
      "Complete the full 6-month medication course",
      "Cover your mouth when coughing",
      "Keep rooms well ventilated",
      "Get regular chest X-rays"
    ],
    "doc": "Pulmonologist",
    "sev": "High"
  },
  "Common Cold": {
    "desc": "A mild viral infection of the nose and throat, very common.",
    "prec": [
      "Rest and drink warm fluids",
      "Gargle with warm salt water",
      "Use a humidifier for comfort",
      "Wash your hands frequently"
    ],
    "doc": "General Physician",
    "sev": "Low"
  },
  "Pneumonia": {
    "desc": "An infection that inflames the air sacs in one or both lungs.",
    "prec": [
      "Complete the full antibiotic course",
      "Rest completely until recovered",
      "Stay well hydrated",
      "Use a humidifier to ease breathing"
    ],
    "doc": "Pulmonologist",
    "sev": "High"
  },
  "Dimorphic hemmorhoids(piles)": {
    "desc": "Swollen blood vessels in the rectal area causing discomfort or bleeding.",
    "prec": [
      "Eat a high-fiber diet",
      "Drink plenty of water",
      "Avoid straining during bowel movements",
      "Take warm sitz baths"
    ],
    "doc": "Proctologist",
    "sev": "Medium"
  },
  "Heart attack": {
    "desc": "A serious emergency where blood flow to the heart is suddenly blocked.",
    "prec": [
      "Call emergency services immediately",
      "Chew an aspirin if available",
      "Rest in a comfortable position",
      "Do not attempt to drive yourself"
    ],
    "doc": "Cardiologist",
    "sev": "Critical"
  },
  "Varicose veins": {
    "desc": "Enlarged, swollen, and twisted veins that usually appear in the legs.",
    "prec": [
      "Elevate your legs when resting",
      "Avoid standing for long periods",
      "Wear compression stockings",
      "Walk regularly to improve circulation"
    ],
    "doc": "Vascular Surgeon",
    "sev": "Low"
  },
  "Hypothyroidism": {
    "desc": "A condition where the thyroid gland does not produce enough hormones.",
    "prec": [
      "Take your thyroid medication daily",
      "Get regular blood tests",
      "Maintain a healthy weight",
      "Include iodine-rich foods in your diet"
    ],
    "doc": "Endocrinologist",
    "sev": "Medium"
  },
  "Hyperthyroidism": {
    "desc": "A condition where the thyroid gland produces too much hormone.",
    "prec": [
      "Take medication exactly as prescribed",
      "Avoid excess iodine in your diet",
      "Monitor your heart rate",
      "Reduce caffeine intake"
    ],
    "doc": "Endocrinologist",
    "sev": "Medium"
  },
  "Hypoglycemia": {
    "desc": "A condition where blood sugar drops below normal levels.",
    "prec": [
      "Carry glucose tablets or candy",
      "Eat regular meals and snacks",
      "Monitor your blood sugar",
      "Wear a medical alert bracelet"
    ],
    "doc": "Endocrinologist",
    "sev": "Medium"
  },
  "Osteoarthristis": {
    "desc": "A degenerative joint disease caused by the breakdown of joint cartilage.",
    "prec": [
      "Do gentle, low-impact exercises",
      "Maintain a healthy body weight",
      "Use hot or cold compresses",
      "Wear supportive, comfortable shoes"
    ],
    "doc": "Orthopedic",
    "sev": "Medium"
  },
  "Arthritis": {
    "desc": "Inflammation of one or more joints, causing pain and stiffness.",
    "prec": [
      "Stay active with gentle exercises",
      "Apply heat or cold to affected joints",
      "Manage your weight",
      "Take prescribed medication on schedule"
    ],
    "doc": "Rheumatologist",
    "sev": "Medium"
  },
  "(vertigo) Paroymsal  Positional Vertigo": {
    "desc": "Brief episodes of dizziness triggered by changes in head position.",
    "prec": [
      "Move slowly when changing positions",
      "Avoid sudden head movements",
      "Sleep with your head slightly elevated",
      "Do not drive during dizzy spells"
    ],
    "doc": "ENT Specialist",
    "sev": "Low"
  },
  "Acne": {
    "desc": "A skin condition that occurs when hair follicles become clogged with oil and dead skin.",
    "prec": [
      "Wash your face twice daily",
      "Avoid touching or picking at your face",
      "Use oil-free, non-comedogenic products",
      "Do not squeeze or pop pimples"
    ],
    "doc": "Dermatologist",
    "sev": "Low"
  },
  "Urinary tract infection": {
    "desc": "An infection in any part of the urinary system, most commonly the bladder.",
    "prec": [
      "Drink plenty of water daily",
      "Urinate frequently and fully",
      "Wipe from front to back",
      "Avoid irritating feminine products"
    ],
    "doc": "Urologist",
    "sev": "Low"
  },
  "Psoriasis": {
    "desc": "A chronic autoimmune condition causing rapid buildup of skin cells with thick scales.",
    "prec": [
      "Keep your skin well moisturized",
      "Identify and avoid your triggers",
      "Use prescribed medicated creams",
      "Manage stress effectively"
    ],
    "doc": "Dermatologist",
    "sev": "Medium"
  },
  "Impetigo": {
    "desc": "A highly contagious skin infection that produces red sores or blisters.",
    "prec": [
      "Keep the affected area clean",
      "Do not share towels or clothing",
      "Complete the full antibiotic course",
      "Wash your hands frequently"
    ],
    "doc": "Dermatologist",
    "sev": "Low"
  }
}

def get_disease_display(raw):
    return DISEASE_DISPLAY_MAP.get(raw, raw.strip().title())

def get_info(raw):
    return DISEASE_INFO.get(raw, {})

SYSTEM_PROMPT = (
    "You are a helpful AI health assistant embedded in a disease prediction web app. "
    "Answer questions about symptoms, diseases, precautions, diagnoses, treatments, "
    "diet, exercise, and general health tips concisely and accurately. "
    "Always remind users that this is for educational purposes only and they should "
    "consult a healthcare professional for actual diagnosis and treatment. "
    "Keep responses under 300 words."
)

def get_chatbot_response(message):
    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": HF_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
        }
        resp = http_requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"AI service unavailable ({type(e).__name__}). Please try again later."


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session: return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session: return redirect(url_for('login'))
        users = load_users()
        user = users.get(session['user_id'])
        if not user or user.get('role') != 'admin':
            flash('Admin access required.', 'error')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated

def do_predict(symptoms_selected):
    vec = [0] * len(SYMPTOMS)
    for s in symptoms_selected:
        for i, sym in enumerate(SYMPTOMS):
            if sym.strip().lower() == s.strip().lower():
                vec[i] = 1
                break
    df = pd.DataFrame([vec], columns=SYMPTOMS)
    return model.predict(df)[0]


# ==============================================================
# ROUTES
# ==============================================================
@app.route('/')
def home():
    from flask import session
    cu = None
    if 'user_id' in session:
        users = load_users()
        cu = users.get(session['user_id'])
    return render_home_page(cu)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        users = load_users()
        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_register_page()
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_register_page()
        if len(password) < 4:
            flash('Password must be at least 4 characters.', 'error')
            return render_register_page()
        for u in users.values():
            if u['username'].lower() == username.lower():
                flash('Username already taken.', 'error')
                return render_register_page()
        uid = str(uuid.uuid4())[:8]
        users[uid] = {'id': uid, 'username': username, 'email': email,
                       'password': generate_password_hash(password),
                       'role': 'user', 'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
                       'predictions': []}
        save_users(users)
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_register_page()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        users = load_users()
        for u in users.values():
            if u['username'].lower() == username.lower() and check_password_hash(u['password'], password):
                session['user_id'] = u['id']
                session['username'] = u['username']
                session['role'] = u['role']
                flash('Welcome back, ' + u['username'] + '!', 'success')
                if u['role'] == 'admin':
                    return redirect(url_for('admin_dash'))
                return redirect(url_for('user_dash'))
        flash('Invalid username or password.', 'error')
    return render_login_page()

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def user_dash():
    users = load_users()
    me = users.get(session['user_id'])
    if not me:
        session.clear()
        return redirect(url_for('login'))
    last_result = session.pop('last_result', None)
    return render_dashboard_page(me, last_result=last_result)

@app.route('/predict', methods=['POST'])
@login_required
def user_predict():
    selected = request.form.getlist('symptoms')
    if not selected:
        flash('Select at least one symptom.', 'warning')
        return redirect(url_for('user_dash'))
    raw = do_predict(selected)
    display = get_disease_display(raw)
    info = get_info(raw)
    users = load_users()
    user = users.get(session['user_id'])
    if user:
        entry = {'symptoms': selected, 'predicted_disease': display,
                 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                 'precautions': info.get('prec', ['Consult a doctor']),
                 'doctor': info.get('doc', 'General Physician'),
                 'severity': info.get('sev', 'Unknown')}
        user['predictions'].append(entry)
        save_users(users)
        session['last_result'] = entry
    return redirect(url_for('user_dash'))

@app.route('/disease/<name>')
@login_required
def disease_info(name):
    search = name.replace('_', ' ').lower()
    for raw, display in DISEASE_DISPLAY_MAP.items():
        if display.lower().replace(' ', '_') == name.lower() or raw.lower() == search or display.lower() == search:
            return render_disease_page(display, DISEASE_INFO.get(raw, {}))
    flash('Disease info not found.', 'warning')
    return redirect(url_for('user_dash'))

@app.route('/admin')
@admin_required
def admin_dash():
    users = load_users()
    return render_admin_page(users)

@app.route('/admin/delete_user/<uid>', methods=['POST'])
@admin_required
def admin_delete(uid):
    users = load_users()
    if uid in users and uid != 'admin':
        del users[uid]
        save_users(users)
        flash('User deleted.', 'success')
    return redirect(url_for('admin_dash'))

@app.route('/admin/view_user/<uid>')
@admin_required
def admin_view(uid):
    users = load_users()
    user = users.get(uid)
    if user:
        return render_admin_view_page(user)
    flash('User not found.', 'error')
    return redirect(url_for('admin_dash'))

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.get_json()
    msg = data.get('message', '')
    resp = get_chatbot_response(msg)
    return jsonify({'response': resp})


# ==============================================================
# CSS + RENDERERS
# ==============================================================
CSS_STYLE = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #f0f4f8; color: #1a202c; line-height: 1.6; }
a { text-decoration: none; color: inherit; }
.flash { padding: 12px 20px; margin: 16px auto; max-width: 900px; border-radius: 8px; text-align: center; font-size: 14px; }
.flash.error { background: #fed7d7; color: #9b2c2c; border: 1px solid #fc8181; }
.flash.success { background: #c6f6d5; color: #276749; border: 1px solid #68d391; }
.flash.warning { background: #fefcbf; color: #975a16; border: 1px solid #f6e05e; }
.flash.info { background: #bee3f8; color: #2a4365; border: 1px solid #63b3ed; }
.navbar { background: #fff; border-bottom: 1px solid #e2e8f0; padding: 0 20px; display: flex; align-items: center; justify-content: space-between; height: 56px; }
.navbar .brand { font-size: 18px; font-weight: 700; color: #2b6cb0; }
.navbar .nav-links a { padding: 6px 12px; font-size: 13px; font-weight: 500; color: #4a5568; margin-left: 4px; border-radius: 6px; }
.navbar .nav-links a:hover { background: #edf2f7; }
.navbar .nav-links a.active { background: #ebf8ff; color: #2b6cb0; }
.navbar .nav-links .nav-user { font-size: 13px; color: #718096; margin-left: 8px; border-left: 1px solid #e2e8f0; padding-left: 8px; }
.navbar .nav-links .btn-out { color: #9b2c2c; }
.navbar .nav-links .btn-out:hover { background: #fed7d7; }
.page { max-width: 900px; margin: 24px auto; padding: 0 16px; }
.card { background: #fff; border-radius: 12px; padding: 24px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.card h2 { font-size: 18px; margin-bottom: 14px; color: #2d3748; }
.form-group { margin-bottom: 14px; }
.form-group label { display: block; font-size: 14px; font-weight: 600; margin-bottom: 6px; color: #4a5568; }
.form-group input { width: 100%; padding: 10px 12px; border: 1px solid #cbd5e0; border-radius: 8px; font-size: 14px; outline: none; }
.form-group input:focus { border-color: #4299e1; box-shadow: 0 0 0 3px rgba(66,153,225,0.15); }
.btn { display: inline-block; padding: 10px 18px; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; }
.btn-primary { background: #3182ce; color: #fff; }
.btn-primary:hover { background: #2c5282; }
.btn-danger { background: #e53e3e; color: #fff; }
.btn-secondary { background: #718096; color: #fff; }
.btn-secondary:hover { background: #4a5568; }
.btn-sm { padding: 5px 10px; font-size: 12px; }
.btn-block { display: block; width: 100%; }
.tag { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
.tag-Low { background: #c6f6d5; color: #276749; }
.tag-Medium { background: #fefcbf; color: #975a16; }
.tag-High { background: #fed7d7; color: #9b2c2c; }
.tag-Critical { background: #9b2c2c; color: #fff; }
.sym-opt { display: flex; align-items: center; padding: 6px 10px; border-radius: 6px; font-size: 13px; cursor: pointer; }
.sym-opt:hover { background: #edf2f7; }
.sym-opt .sym-chk { margin-right: 6px; }
.sym-opt.sel { background: #ebf8ff; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 700; }
.badge-admin { background: #9b2c2c; color: #fff; }
.badge-user { background: #bee3f8; color: #2a4365; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { background: #f7fafc; padding: 10px 12px; text-align: left; font-weight: 600; color: #4a5568; border-bottom: 2px solid #e2e8f0; }
td { padding: 10px 12px; border-bottom: 1px solid #edf2f7; }
tr:hover td { background: #f7fafc; }
.auth-wrap { display: flex; min-height: 100vh; align-items: center; justify-content: center; background: #f0f4f8; }
.auth-card { background: #fff; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); width: 400px; }
.auth-hdr { background: linear-gradient(135deg, #3182ce, #63b3ed); color: #fff; padding: 28px; text-align: center; border-radius: 16px 16px 0 0; }
.auth-hdr h1 { font-size: 22px; }
.auth-hdr p { opacity: 0.85; font-size: 13px; }
.auth-body { padding: 24px; }
.auth-ftr { text-align: center; padding: 0 24px 20px; font-size: 14px; color: #718096; }
.auth-ftr a { color: #3182ce; font-weight: 600; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 16px; }
.stat-card { background: #f7fafc; border-radius: 10px; padding: 16px; text-align: center; }
.stat-card .num { font-size: 28px; font-weight: 700; color: #3182ce; }
.stat-card .lbl { font-size: 12px; color: #718096; margin-top: 2px; }
.pred-item { padding: 14px; border-bottom: 1px solid #edf2f7; }
.pred-item:last-child { border-bottom: none; }
.pred-dis { font-size: 16px; font-weight: 700; color: #2d3748; }
.pred-meta { font-size: 12px; color: #718096; margin-top: 2px; }
.pred-sym { font-size: 12px; color: #4a5568; margin-top: 4px; }
.prec-list { list-style: disc; padding-left: 20px; margin-top: 6px; }
.prec-list li { font-size: 12px; color: #4a5568; }
.d-tags { display: flex; flex-wrap: wrap; gap: 4px; }
.d-tag { display: inline-block; padding: 3px 8px; background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 11px; color: #4a5568; }
footer { text-align: center; padding: 20px; color: #a0aec0; font-size: 12px; margin-top: 24px; border-top: 1px solid #e2e8f0; background: #fff; }
@media (max-width: 640px) { .auth-card { width: 92%%; } .card { padding: 16px; } }

.chat-fab { position: fixed; bottom: 24px; right: 24px; width: 60px; height: 60px; border-radius: 50%%; background: linear-gradient(135deg, #3182ce, #4299e1); border: none; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(49,130,206,0.4); transition: transform .2s, box-shadow .2s; z-index: 10000; }
.chat-fab:hover { transform: scale(1.1); box-shadow: 0 6px 20px rgba(49,130,206,0.5); }
.chat-fab svg { width: 24px; height: 24px; fill: #fff; }

.chat-panel { position: fixed; bottom: 96px; right: 24px; width: 380px; max-height: 560px; background: #fff; border-radius: 16px; box-shadow: 0 12px 40px rgba(0,0,0,0.18); display: none; flex-direction: column; z-index: 10001; overflow: hidden; animation: chatIn .25s ease; }
.chat-panel.open { display: flex; }
@keyframes chatIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }

.chat-header { background: linear-gradient(135deg, #2b6cb0, #3182ce); color: #fff; padding: 14px 18px; display: flex; align-items: center; gap: 10px; }
.chat-header-icon { width: 36px; height: 36px; border-radius: 50%%; background: rgba(255,255,255,0.2); display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.chat-header-icon svg { width: 18px; height: 18px; fill: #fff; }
.chat-header-text { flex: 1; }
.chat-header-text strong { font-size: 15px; display: block; }
.chat-header-text small { font-size: 11px; opacity: 0.85; }
.chat-header-close { background: none; border: none; color: #fff; cursor: pointer; padding: 4px; opacity: 0.8; transition: opacity .15s; }
.chat-header-close:hover { opacity: 1; }
.chat-header-close svg { width: 18px; height: 18px; fill: #fff; }

.chat-msgs { flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 6px; background: #f5f7fa; min-height: 100px; max-height: 400px; }
.chat-msg { display: flex; gap: 8px; max-width: 85%%; animation: msgIn .2s ease; }
@keyframes msgIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
.chat-msg.bot { align-self: flex-start; }
.chat-msg.usr { align-self: flex-end; flex-direction: row-reverse; }

.chat-ava { width: 28px; height: 28px; border-radius: 50%%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-size: 14px; }
.chat-msg.bot .chat-ava { background: linear-gradient(135deg, #3182ce, #4299e1); color: #fff; }
.chat-msg.usr .chat-ava { background: #e2e8f0; color: #4a5568; }
.chat-msg.bot .chat-ava svg, .chat-msg.usr .chat-msg.usr .chat-ava svg { width: 14px; height: 14px; fill: currentColor; }

.chat-bubble { padding: 10px 14px; border-radius: 14px; font-size: 13.5px; line-height: 1.55; word-wrap: break-word; }
.chat-msg.bot .chat-bubble { background: #fff; color: #1a202c; border: 1px solid #e2e8f0; border-top-left-radius: 4px; }
.chat-msg.usr .chat-bubble { background: linear-gradient(135deg, #3182ce, #4299e1); color: #fff; border-top-right-radius: 4px; }
.chat-bubble strong { font-weight: 600; }
.chat-bubble em { font-style: italic; }
.chat-bubble ul, .chat-bubble ol { margin: 4px 0 4px 16px; }
.chat-bubble li { margin-bottom: 2px; }
.chat-bubble br { line-height: 1.5; }

.typing-dots { display: flex; gap: 4px; padding: 10px 14px; }
.typing-dots span { width: 7px; height: 7px; border-radius: 50%%; background: #a0aec0; animation: bounce .6s infinite alternate; }
.typing-dots span:nth-child(2) { animation-delay: .2s; }
.typing-dots span:nth-child(3) { animation-delay: .4s; }
@keyframes bounce { to { transform: translateY(-5px); opacity: .6; } }

.chat-bar { display: flex; padding: 10px 12px; gap: 8px; border-top: 1px solid #e2e8f0; background: #fff; }
.chat-bar input { flex: 1; border: 1px solid #cbd5e0; border-radius: 24px; padding: 10px 16px; font-size: 13px; outline: none; transition: border-color .15s; }
.chat-bar input:focus { border-color: #4299e1; box-shadow: 0 0 0 3px rgba(66,153,225,0.15); }
.chat-bar button { border: none; border-radius: 50%%; width: 40px; height: 40px; background: linear-gradient(135deg, #3182ce, #4299e1); cursor: pointer; display: flex; align-items: center; justify-content: center; transition: transform .15s; flex-shrink: 0; }
.chat-bar button:hover { transform: scale(1.05); }
.chat-bar button:active { transform: scale(0.95); }
.chat-bar button svg { width: 16px; height: 16px; fill: #fff; }

.chat-suggestions { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 14px 8px; }
.chat-sugg { background: #ebf8ff; color: #2b6cb0; border: 1px solid #bee3f8; border-radius: 14px; padding: 5px 12px; font-size: 12px; cursor: pointer; transition: background .15s; }
.chat-sugg:hover { background: #bee3f8; }

@media (max-width: 640px) { .chat-panel { width: calc(100vw - 24px); right: 12px; bottom: 84px; } .chat-fab { bottom: 16px; right: 16px; } }
"""


def _head():
    return '<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">' \
           '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">' \
           '<style>' + CSS_STYLE + '</style></head>'


def _fl():
    from flask import get_flashed_messages
    msgs = get_flashed_messages(with_categories=True)
    if msgs:
        return "".join(['<div class="flash %s">%s</div>' % (c, m) for c, m in msgs])
    return ""


def _nav(cu=None, act=""):
    n = '<nav class="navbar"><div class="brand"><i class="fa-solid fa-stethoscope"></i> Health Predictor</div>'
    n += '<div class="nav-links">'
    n += '<a href="/" class="%s">Home</a>' % ("active" if act == "home" else "")
    if cu:
        if cu.get("role") == "admin":
            n += '<a href="/admin" class="%s">Admin</a>' % ("active" if act == "admin" else "")
        else:
            n += '<a href="/dashboard" class="%s">Dashboard</a>' % ("active" if act == "dash" else "")
        n += '<span class="nav-user">%s</span>' % cu.get("username", "")
        n += '<a href="/logout" class="btn-out">Logout</a>'
    else:
        n += '<a href="/login" class="%s">Sign In</a>' % ("active" if act == "login" else "")
        n += '<a href="/register" class="btn btn-primary btn-sm">Sign Up</a>'
    n += '</div></nav>'
    return n
JS_SYM = """function ucnt(){var n=document.querySelectorAll("#symBox input:checked").length;document.getElementById("sct").textContent=n+" selected";}function clrSym(){document.querySelectorAll("#symBox input.sym-chk").forEach(function(c){c.checked=false;c.parentElement.classList.remove("sel");});ucnt();}function initSymHandlers(){document.querySelectorAll("#symBox .sym-chk").forEach(function(c){c.addEventListener("change",function(){if(this.checked){this.parentElement.classList.add("sel");}else{this.parentElement.classList.remove("sel");}ucnt();});});}"""

JS_CHAT = """function toggleChat(){var p=document.getElementById('chatPanel');p.classList.toggle('open');}function askSug(el){document.getElementById('cIn').value=el.textContent;cSend();}function fmtMd(t){return t.replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>').replace(/\\*(.+?)\\*/g,'<em>$1</em>').replace(/\\n/g,'<br>');}function addMsg(txt,who){var d=document.getElementById('cMsgs'),s=document.getElementById('chatSugg');if(s)s.style.display='none';var m=document.createElement('div');m.className='chat-msg '+who;var av=document.createElement('div');av.className='chat-ava';av.innerHTML=who==='bot'?'&#10010;':'&#128100;';var bub=document.createElement('div');bub.className='chat-bubble';bub.innerHTML=who==='bot'?fmtMd(txt):txt.replace(/</g,'&lt;').replace(/\\n/g,'<br>');m.appendChild(av);m.appendChild(bub);d.appendChild(m);d.scrollTop=d.scrollHeight;}function showTyping(){var d=document.getElementById('cMsgs');var t=document.createElement('div');t.className='chat-msg bot';t.id='typing';var av=document.createElement('div');av.className='chat-ava';av.innerHTML='&#10010;';var bub=document.createElement('div');bub.className='chat-bubble';bub.innerHTML='<div class="typing-dots"><span></span><span></span><span></span></div>';t.appendChild(av);t.appendChild(bub);d.appendChild(t);d.scrollTop=d.scrollHeight;}function hideTyping(){var t=document.getElementById('typing');if(t)t.remove();}function cSend(){var m=document.getElementById('cIn'),t=m.value.trim();if(!t)return;if(m.disabled)return;m.disabled=true;addMsg(t,'usr');m.value="";showTyping();fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t})}).then(function(r){return r.json()}).then(function(r){hideTyping();addMsg(r.response,'bot');m.disabled=false;m.focus();}).catch(function(){hideTyping();addMsg('Sorry, the AI service is unavailable. Please try again later.','bot');m.disabled=false;});}"""

def _sym_opts():
    o = ""
    for s in SYMPTOMS:
        lbl = s.replace("_", " ").title()
        o += '<label class="sym-opt">'
        o += '<input type="checkbox" name="symptoms" value="' + s + '" class="sym-chk">' + lbl + '</label>'
    return o

def _chat_box():
    c  = '<button class="chat-fab" onclick="toggleChat()" aria-label="Open chat" id="chatFab">'
    c += '<svg viewBox="0 0 24 24"><path d="M12 2C6.477 2 2 6.477 2 12c0 1.322.261 2.585.724 3.744L2 19l3.256-.724A9.97 9.97 0 0 0 9 22h3c5.523 0 10-4.477 10-10S17.523 2 12 2z"/></svg></button>'
    c += '<div class="chat-panel" id="chatPanel">'
    c += '<div class="chat-header">'
    c += '<div class="chat-header-icon"><svg viewBox="0 0 24 24"><path d="M12 2C6.477 2 2 6.477 2 12c0 1.322.261 2.585.724 3.744L2 19l3.256-.724A9.97 9.97 0 0 0 9 22h3c5.523 0 10-4.477 10-10S17.523 2 12 2z"/></svg></div>'
    c += '<div class="chat-header-text"><strong>Health Assistant</strong><small>Online</small></div>'
    c += '<button class="chat-header-close" onclick="toggleChat()"><svg viewBox="0 0 24 24"><path d="M18.3 5.71a1 1 0 0 0-1.41 0L12 10.59 7.11 5.71A1 1 0 0 0 5.7 7.11L10.59 12l-4.88 4.89a1 1 0 1 0 1.41 1.41L12 13.41l4.89 4.88a1 1 0 0 0 1.41-1.41L13.41 12l4.88-4.89a1 1 0 0 0 0-1.4z"/></svg></button>'
    c += '</div>'
    c += '<div class="chat-msgs" id="cMsgs">'
    c += '<div class="chat-msg bot"><div class="chat-ava">&#10010;</div><div class="chat-bubble">Hi! Ask me about any symptom, disease, precaution, or health tip.</div></div>'
    c += '</div>'
    c += '<div class="chat-suggestions" id="chatSugg">'
    c += '<span class="chat-sugg" onclick="askSug(this)">Headache</span>'
    c += '<span class="chat-sugg" onclick="askSug(this)">Diabetes</span>'
    c += '<span class="chat-sugg" onclick="askSug(this)">Fever remedies</span>'
    c += '<span class="chat-sugg" onclick="askSug(this)">Healthy diet tips</span>'
    c += '</div>'
    c += '<div class="chat-bar">'
    c += '<input type="text" id="cIn" placeholder="Type your question..." autocomplete="off" onkeydown="if(event.key===\'Enter\')cSend()">'
    c += '<button onclick="cSend()" aria-label="Send"><svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg></button>'
    c += '</div></div>'
    c += '<script>' + JS_CHAT + '</script>'
    return c

def _footer():
    return '<footer>AI Health Symptoms Checker &amp; Disease Predictor - Educational purposes only. Consult a healthcare professional.</footer>'

def render_register_page():
    h = '<!DOCTYPE html><html>' + _head()
    h += '<body class="auth-wrap"><div>'
    h += '<a href="/" style="color:#3182ce;font-weight:600;margin-bottom:12px;display:inline-block;">Back</a>'
    h += '<div class="auth-card"><div class="auth-hdr"><h1>Create Account</h1><p>Join to track your health predictions</p></div>'
    h += '<div class="auth-body"><form method="POST" action="/register">'
    h += '<div class="form-group"><label>Username</label><input type="text" name="username" required placeholder="Choose a username"></div>'
    h += '<div class="form-group"><label>Email</label><input type="email" name="email" required placeholder="you@example.com"></div>'
    h += '<div class="form-group"><label>Password</label><input type="password" name="password" required placeholder="Min 4 characters"></div>'
    h += '<div class="form-group"><label>Confirm Password</label><input type="password" name="confirm_password" required placeholder="Repeat password"></div>'
    h += _fl() if _fl() else ''
    h += '<button type="submit" class="btn btn-primary btn-block">Create Account</button></form></div>'
    h += '<div class="auth-ftr">Already have an account? <a href="/login">Sign in</a></div>'
    h += '</div></div></body></html>'
    return h

def render_login_page():
    h = '<!DOCTYPE html><html>' + _head()
    h += '<body class="auth-wrap"><div>'
    h += '<a href="/" style="color:#3182ce;font-weight:600;margin-bottom:12px;display:inline-block;">Back</a>'
    h += '<div class="auth-card"><div class="auth-hdr"><h1>Welcome Back</h1><p>Sign in to your account</p></div>'
    h += '<div class="auth-body"><form method="POST" action="/login">'
    h += '<div class="form-group"><label>Username</label><input type="text" name="username" required placeholder="Enter username"></div>'
    h += '<div class="form-group"><label>Password</label><input type="password" name="password" required placeholder="Enter password"></div>'
    h += _fl() if _fl() else ''
    h += '<button type="submit" class="btn btn-primary btn-block">Sign In</button></form></div>'
    h += '<div class="auth-ftr">No account? <a href="/register">Sign up</a></div>'
    h += '</div></div></body></html>'
    return h

def render_home_page(cu):
    last = session.pop("last_result", None)
    r_html = ""
    if last:
        r_html += '<div class="card" style="border-left:4px solid #38a169;"><h2 style="color:#2f855a;">Prediction Result</h2>'
        r_html += '<p style="font-size:20px;font-weight:700;">%s %s</p>' % (last.get("predicted_disease", ""), _sev_tag(last.get("severity", "")))
        r_html += '<p style="font-size:13px;color:#718096;">Doctor: %s | Symptoms: %s</p>' % (last.get("doctor", ""), ", ".join([x.replace("_", " ").title() for x in last.get("symptoms", [])]))
        r_html += '<p style="font-size:13px;margin-top:6px;"><strong>Precautions:</strong></p><ul class="prec-list">'
        for pc in last.get("precautions", []):
            r_html += "<li>" + pc + "</li>"
        r_html += "</ul></div>"

    h = '<!DOCTYPE html><html>' + _head() + '<body>' + _nav(cu, "home") + _fl()
    h += '<div class="page">'
    h += '<div style="text-align:center;padding:28px 20px 20px;">'
    h += '<div style="font-size:36px;color:#3182ce;"><i class="fa-solid fa-stethoscope"></i></div>'
    h += '<h1 style="font-size:26px;margin:8px 0 4px;">AI Health Symptoms Checker</h1>'
    h += '<p style="color:#718096;font-size:14px;">Select your symptoms to get a disease prediction.</p>'
    h += '<div style="margin-top:8px;display:inline-flex;gap:6px;">'
    h += '<span style="background:rgba(49,130,206,0.1);padding:4px 12px;border-radius:14px;font-size:12px;color:#2b6cb0;">132 Symptoms</span>'
    h += '<span style="background:rgba(49,130,206,0.1);padding:4px 12px;border-radius:14px;font-size:12px;color:#2b6cb0;">41 Diseases</span>'
    h += '<span style="background:rgba(49,130,206,0.1);padding:4px 12px;border-radius:14px;font-size:12px;color:#2b6cb0;">Decision Tree</span>'
    h += '</div></div>'

    h += r_html
    h += '<div class="card"><h2>Select Your Symptoms</h2>'
    h += '<form method="POST" action="/predict" id="predForm">'
    h += '<div class="form-group"><input type="text" id="symS" placeholder="Search symptoms..." autocomplete="off"></div>'
    h += '<div id="symBox" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:2px;max-height:280px;overflow-y:auto;padding:4px;">'
    h += _sym_opts()
    h += '</div>'
    h += '<div style="display:flex;gap:10px;justify-content:center;margin-top:12px;align-items:center;">'
    h += '<span id="sct" style="font-size:13px;color:#718096;">0 selected</span>'
    h += '<button type="submit" class="btn btn-primary">Predict Disease</button>'
    h += '<button type="button" class="btn btn-secondary" onclick="clrSym()">Clear</button>'
    h += '</div></form></div>'

    if not cu:
        h += '<div class="card" style="text-align:center;color:#718096;font-size:14px;">'
        h += 'Want to save your prediction history? <a href="/register" style="color:#3182ce;font-weight:600;">Create an account</a> or <a href="/login" style="color:#3182ce;font-weight:600;">Sign in</a>.'
        h += '</div>'

    h += '</div>' + _footer()
    h += '<script>' + JS_SYM + 'initSymHandlers();ucnt();document.getElementById("symS").addEventListener("input",function(e){var q=e.target.value.toLowerCase();document.querySelectorAll("#symBox .sym-opt").forEach(function(l){l.style.display=l.textContent.toLowerCase().includes(q)?"flex":"none";});});initChat();'
    h += 'function initChat(){' + JS_CHAT + '}'
    h += '</script></body></html>'
    return h

def render_dashboard_page(me, last_result=None):
    preds = me.get("predictions", [])

    def build_result_card():
        if not last_result:
            return ""
        card = '<div class="card" style="border:2px solid #4299e1;background:#ebf8ff;">'
        card += '<h2 style="color:#2b6cb0;font-size:18px;"><i class="fa-solid fa-check-circle" style="color:#38a169;"></i> Prediction Result</h2>'
        card += '<div class="pred-dis" style="font-size:20px;margin-top:10px;color:#2d3748;">%s %s</div>' % (last_result.get("predicted_disease", ""), '<span class="tag tag-%s">%s</span>' % (last_result.get("severity", ""), last_result.get("severity", "")))
        if last_result.get("doctor"):
            card += '<p style="font-size:13px;color:#4a5568;margin-top:6px;"><strong>Recommended Doctor:</strong> %s</p>' % last_result["doctor"]

        prec = last_result.get("precautions", [])
        if prec:
            card += '<div style="margin-top:12px;"><strong style="font-size:14px;color:#2d3748;">Precautions &amp; Remedies</strong>'
            card += '<ul class="prec-list" style="margin-top:6px;">'
            for p in prec:
                card += "<li>%s</li>" % p
            card += "</ul></div>"

        sym_text = ", ".join([s.replace("_", " ").title() for s in last_result.get("symptoms", [])])
        card += '<div class="pred-sym" style="margin-top:8px;">Symptoms: %s</div>' % sym_text
        card += '<div class="pred-meta" style="margin-top:4px;">Date: %s</div>' % last_result.get("timestamp", "")
        card += '</div>'
        return card

    def build_preds():
        if not preds:
            return '<p style="text-align:center;color:#a0aec0;padding:24px;">No predictions yet. Select symptoms above.</p>'
        items = ""
        for p in reversed(preds):
            d = p.get("predicted_disease", "")
            sev = p.get("severity", "")
            items += '<div class="pred-item">'
            items += '<div class="pred-dis">%s %s</div>' % (d, '<span class="tag tag-%s">%s</span>' % (sev, sev))
            items += '<div class="pred-meta">Date: %s | Doctor: %s</div>' % (p.get("timestamp", ""), p.get("doctor", ""))
            items += '<div class="pred-sym">Symptoms: %s</div>' % ", ".join([s.replace("_", " ").title() for s in p.get("symptoms", [])])
            prec = p.get("precautions", [])
            if prec:
                items += '<ul class="prec-list">'
                for pc in prec:
                    items += "<li>" + pc + "</li>"
                items += '</ul>'
            items += '</div>'
        return items

    def build_d_links():
        links = ""
        for raw, display in sorted(DISEASE_DISPLAY_MAP.items(), key=lambda x: x[1]):
            links += '<a href="/disease/%s" class="d-tag">%s</a>' % (display.replace(" ", "_"), display)
        return links

    h = '<!DOCTYPE html><html>' + _head() + '<body>' + _nav(me, "dash") + _fl()
    h += '<div class="page">'
    h += '<h1 style="font-size:22px;margin-bottom:14px;">Welcome, %s</h1>' % me.get("username", "")
    h += '<div class="stat-grid">'
    h += '<div class="stat-card"><div class="num">%d</div><div class="lbl">Predictions</div></div>' % len(preds)
    h += '<div class="stat-card"><div class="num">%s</div><div class="lbl">Joined</div></div>' % me.get("created_at", "")[:4]
    h += '<div class="stat-card"><div class="num"><span class="badge badge-user">User</span></div><div class="lbl">Role</div></div>'
    h += '</div>'

    h += build_result_card()

    h += '<div class="card"><h2>Check Symptoms</h2>'
    h += '<form method="POST" action="/predict">'
    h += '<div class="form-group"><input type="text" id="symS" placeholder="Search symptoms..." autocomplete="off"></div>'
    h += '<div id="symBox" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:2px;max-height:240px;overflow-y:auto;padding:4px;">'
    h += _sym_opts()
    h += '</div>'
    h += '<div style="display:flex;gap:10px;justify-content:center;margin-top:12px;align-items:center;">'
    h += '<span id="sct" style="font-size:13px;color:#718096;">0 selected</span>'
    h += '<button type="submit" class="btn btn-primary">Predict</button>'
    h += '<button type="button" class="btn btn-secondary" onclick="clrSym()">Clear</button>'
    h += '</div></form></div>'

    h += '<div class="card"><h2>Prediction History</h2>' + build_preds() + '</div>'

    h += '<div class="card"><h2>Disease Information</h2>'
    h += '<p style="font-size:13px;color:#718096;margin-bottom:6px;">Click a disease to learn about precautions and recommended doctors.</p>'
    h += '<div style="max-height:180px;overflow-y:auto;padding:4px;" class="d-tags">' + build_d_links() + '</div></div>'

    h += _chat_box()
    h += '</div>' + _footer()
    h += '<script>' + JS_SYM + 'initSymHandlers();ucnt();document.getElementById("symS").addEventListener("input",function(e){var q=e.target.value.toLowerCase();document.querySelectorAll("#symBox .sym-opt").forEach(function(l){l.style.display=l.textContent.toLowerCase().includes(q)?"flex":"none";});});</script>'
    h += '<script>function initChat(){' + JS_CHAT + '}initChat();</script></body></html>'
    return h

def render_disease_page(display, info):
    from flask import session
    cu = None
    if "user_id" in session:
        users = load_users()
        cu = users.get(session["user_id"])
    h = '<!DOCTYPE html><html>' + _head() + '<body>' + _nav(cu, "dash") + _fl()
    h += '<div class="page"><div class="card">'
    h += '<h2 style="font-size:22px;">%s</h2>' % display
    if info.get("sev"):
        h += '<span class="tag tag-%s">%s</span>' % (info["sev"], info["sev"])
    if info.get("desc"):
        h += '<p style="margin-top:10px;color:#4a5568;">%s</p>' % info["desc"]
    h += '</div><div class="stat-grid">'
    h += '<div class="card"><h3>Precautions</h3><ul class="prec-list">'
    for p in info.get("prec", []):
        h += "<li>" + p + "</li>"
    h += "</ul></div>"
    h += '<div class="card"><h3>Doctor</h3><p style="font-size:16px;font-weight:700;margin-top:8px;">%s</p></div>' % info.get("doc", "")
    h += '</div><p style="text-align:center;margin-top:12px;"><a href="/dashboard" class="btn btn-secondary">Back</a></p>'
    h += '</div>' + _footer() + '</body></html>'
    return h

def render_admin_page(users):
    from flask import session
    cu = users.get(session["user_id"])
    user_list = [v for k, v in users.items() if v.get("role") != "admin"]
    total_p = sum(len(v.get("predictions", [])) for v in users.values())
    h = '<!DOCTYPE html><html>' + _head() + '<body>' + _nav(cu, "admin") + _fl()
    h += '<div class="page">'
    h += '<h1 style="font-size:22px;margin-bottom:14px;">Admin Dashboard</h1>'
    h += '<div class="stat-grid">'
    h += '<div class="stat-card"><div class="num">%d</div><div class="lbl">Users</div></div>' % len(user_list)
    h += '<div class="stat-card"><div class="num">%d</div><div class="lbl">Predictions</div></div>' % total_p
    h += '<div class="stat-card"><div class="num"><span class="badge badge-admin">Admin</span></div><div class="lbl">Role</div></div>'
    h += '</div>'
    h += '<div class="card"><h2>Registered Users</h2>'
    if user_list:
        h += '<div style="overflow-x:auto;"><table><thead><tr><th>User</th><th>Email</th><th>Role</th><th>Joined</th><th>Pred.</th><th>Actions</th></tr></thead><tbody>'
        for u in user_list:
            h += '<tr><td><strong>%s</strong></td>' % u.get("username", "")
            h += '<td>%s</td>' % u.get("email", "")
            h += '<td><span class="badge badge-%s">%s</span></td>' % (u.get("role", "user"), u.get("role", "user"))
            h += '<td>%s</td>' % u.get("created_at", "")
            h += '<td>%d</td>' % len(u.get("predictions", []))
            h += '<td><a href="/admin/view_user/%s" class="btn btn-primary btn-sm">View</a> ' % u.get("id", "")
            h += '<form method="POST" action="/admin/delete_user/%s" style="display:inline;" onsubmit="return confirm(\'Delete?\');">' % u.get("id", "")
            h += '<button type="submit" class="btn btn-danger btn-sm">Delete</button></form></td></tr>'
        h += '</tbody></table></div>'
    else:
        h += '<p style="text-align:center;color:#a0aec0;padding:20px;">No registered users.</p>'
    h += '</div></div>' + _footer() + '</body></html>'
    return h

def render_admin_view_page(user):
    from flask import session
    users = load_users()
    cu = users.get(session["user_id"])
    preds = user.get("predictions", [])
    ph = ""
    if preds:
        for p in preds:
            ph += '<div class="pred-item">'
            ph += '<div class="pred-dis">%s %s</div>' % (p.get("predicted_disease", ""), '<span class="tag tag-%s">%s</span>' % (p.get("severity", ""), p.get("severity", "")))
            ph += '<div class="pred-meta">Date: %s | Doctor: %s</div>' % (p.get("timestamp", ""), p.get("doctor", ""))
            ph += '<div class="pred-sym">Symptoms: %s</div>' % ", ".join([s.replace("_", " ").title() for s in p.get("symptoms", [])])
            prec = p.get("precautions", [])
            if prec:
                ph += '<ul class="prec-list">'
                for pc in prec:
                    ph += "<li>" + pc + "</li>"
                ph += '</ul>'
            ph += '</div>'
    else:
        ph = '<p style="text-align:center;color:#a0aec0;padding:16px;">No predictions.</p>'

    h = '<!DOCTYPE html><html>' + _head() + '<body>' + _nav(cu, "admin") + _fl()
    h += '<div class="page">'
    h += '<h1 style="font-size:22px;margin-bottom:14px;">User: %s</h1>' % user.get("username", "")
    h += '<div class="stat-grid">'
    h += '<div class="card"><h3>Info</h3>'
    h += '<p><strong>Username:</strong> %s</p>' % user.get("username", "")
    h += '<p><strong>Email:</strong> %s</p>' % user.get("email", "")
    h += '<p><strong>Role:</strong> %s</p>' % user.get("role", "")
    h += '<p><strong>Joined:</strong> %s</p></div>' % user.get("created_at", "")
    h += '<div class="card"><h3>Stats</h3><p><strong>Predictions:</strong> %d</p></div>' % len(preds)
    h += '</div>'
    h += '<div class="card" style="margin-top:12px;"><h2>Prediction Records</h2>' + ph + '</div>'
    h += '<p style="text-align:center;margin-top:12px;"><a href="/admin" class="btn btn-secondary">Back to Admin</a></p>'
    h += '</div>' + _footer() + '</body></html>'
    return h


if __name__ == "__main__":
    print("Starting AI Health Symptoms Checker & Disease Predictor...")
    print("Model: decision_tree.joblib | DB: data/users.json")
    print("Admin login: admin / admin123")
    print("Visit: http://127.0.0.1:8080")
    app.run(debug=True, host="127.0.0.1", port=8080)
