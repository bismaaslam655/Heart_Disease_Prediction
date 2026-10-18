import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# ---------- Look and feel: light page, crimson and rose accents ----------
st.markdown(
    """
    <style>
    .stApp { background: #fdf3f4; }
    .stApp p, .stApp label, .stApp span, .stApp li { color: #3b1a22; }
    .hero {
        background: linear-gradient(100deg, #8e1b3a 0%, #d6455d 100%);
        border-radius: 16px; padding: 26px 30px; margin-bottom: 16px;
    }
    .hero h1 { color: #ffffff !important; margin: 0; font-size: 2rem; font-weight: 700; }
    .hero p  { color: #ffe3e8 !important; margin: 6px 0 0 0; font-size: 1rem; }
    .notice {
        background: #fff4d6; border: 1px solid #f0c75e; border-left: 6px solid #e0a100;
        border-radius: 10px; padding: 12px 16px; margin-bottom: 18px; color: #5c4300;
    }
    .notice, .notice b { color: #5c4300 !important; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff; border: 1px solid #f1d3d8; border-radius: 14px;
    }
    .q { color: #8e1b3a; font-weight: 700; font-size: 0.95rem; margin-bottom: 2px; }
    .q b {
        display: inline-block; background: #d6455d; color: #fff; width: 24px; height: 24px;
        line-height: 24px; text-align: center; border-radius: 50%; margin-right: 8px; font-size: 0.8rem;
    }
    .stNumberInput input, .stNumberInput [data-baseweb="input"], .stNumberInput [data-baseweb="base-input"] {
        background: #fff7f8 !important; color: #3b1a22 !important;
    }
    .summary { background: #ffffff; border: 1px solid #f1d3d8; border-radius: 14px; padding: 18px 20px; }
    .summary h4 { color: #8e1b3a; margin: 0 0 10px 0; }
    .row { display: flex; justify-content: space-between; gap: 12px; padding: 7px 0;
           border-bottom: 1px solid #f7e4e7; color: #3b1a22; }
    .row:last-child { border-bottom: none; }
    .row span:last-child { font-weight: 700; color: #d6455d; text-align: right; }
    .stButton > button {
        width: 100%; background: #8e1b3a; color: #ffffff; border: none; border-radius: 10px;
        padding: 0.7rem 1rem; font-weight: 600; font-size: 1rem; margin-top: 10px;
    }
    .stButton > button:hover { background: #d6455d; color: #ffffff; }
    .verdict { border-radius: 14px; padding: 20px 10px; text-align: center; margin-top: 14px;
               font-size: 1.5rem; font-weight: 800; letter-spacing: 1.5px; }
    .verdict.yes { background: #fde8e8; color: #a32727 !important; border: 2px solid #ee8c8c; }
    .verdict.no  { background: #e3f6ea; color: #17753a !important; border: 2px solid #5cc487; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    # Saved pipeline = preprocessing + SVC, exactly as trained in the notebook
    with open(Path(__file__).parent / "heart_disease_model.pkl", "rb") as f:
        return pickle.load(f)


st.markdown(
    """
    <div class="hero">
        <h1>❤️ Heart Disease Prediction</h1>
        <p>Enter four details and the model will indicate whether heart disease is likely.</p>
    </div>
    <div class="notice">
        <b>Educational use only.</b> This app is a university learning project and is
        <b>NOT a medical diagnosis</b>. Please consult a doctor for any health concern.
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model()
except Exception as e:
    st.error(f"The trained model could not be loaded: {e}")
    st.stop()

# Exactly four inputs (labels -> the numeric codes used in the dataset)
SEX = {"Female": 0, "Male": 1}
CP = {
    "Typical angina": 1,
    "Atypical angina": 2,
    "Non-anginal pain": 3,
    "Asymptomatic": 4,
}
EXANG = {"No": 0, "Yes": 1}

left, right = st.columns([3, 2])

with left:
    with st.container(border=True):
        st.markdown('<div class="q"><b>1</b>Sex</div>', unsafe_allow_html=True)
        sex_label = st.radio("Sex", list(SEX), horizontal=True, label_visibility="collapsed")
    with st.container(border=True):
        st.markdown('<div class="q"><b>2</b>Chest pain type</div>', unsafe_allow_html=True)
        cp_label = st.radio("Chest pain type", list(CP), label_visibility="collapsed")
    with st.container(border=True):
        st.markdown('<div class="q"><b>3</b>Maximum heart rate achieved (beats per minute)</div>', unsafe_allow_html=True)
        thalach = st.number_input(
            "Maximum heart rate achieved (beats per minute)",
            min_value=60, max_value=220, value=150, step=1, label_visibility="collapsed",
        )
    with st.container(border=True):
        st.markdown('<div class="q"><b>4</b>Exercise-induced angina</div>', unsafe_allow_html=True)
        exang_label = st.radio("Exercise-induced angina", list(EXANG), horizontal=True, label_visibility="collapsed")

with right:
    st.markdown(
        f"""
        <div class="summary">
            <h4>Patient summary</h4>
            <div class="row"><span>Sex</span><span>{sex_label}</span></div>
            <div class="row"><span>Chest pain</span><span>{cp_label}</span></div>
            <div class="row"><span>Max heart rate</span><span>{int(thalach)} bpm</span></div>
            <div class="row"><span>Exercise angina</span><span>{exang_label}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if thalach < 71 or thalach > 202:
        st.info("This heart rate is outside the range (71-202) seen in the training data, so the prediction may be less reliable.")

    if st.button("Predict"):
        try:
            # Same four columns, same order as training: sex, cp, thalach, exang
            user_input = pd.DataFrame({
                "sex": [SEX[sex_label]],
                "cp": [CP[cp_label]],
                "thalach": [int(thalach)],
                "exang": [EXANG[exang_label]],
            })
            prediction = model.predict(user_input)[0]
            if prediction == 1:
                st.markdown('<div class="verdict yes">DISEASE INDICATED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict no">NO DISEASE INDICATED</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
