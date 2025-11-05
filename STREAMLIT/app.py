import streamlit as st
import pandas as pd
import joblib
import json
import warnings
from streamlit_option_menu import option_menu
import requests  # For Lottie
from streamlit_lottie import st_lottie  # For Lottie
from streamlit_extras.add_vertical_space import add_vertical_space # Import for home page
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

warnings.filterwarnings('ignore')

st.markdown("""
    <style>
    header[data-testid="stHeader"] { display: none; }
    div[data-testid="stToolbar"] { display: none !important; }
    button[kind="header"] { display: none !important; }
    div.block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="HD Predictor",
    page_icon="🧠",
    layout="wide"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="st-"], [class*="css-"] {
    font-family: 'Inter', sans-serif;
    color: #111111;
}

[data-testid="stAppViewContainer"] {
    background-color: #F0F2F5;
}

[data-testid="stHeader"] {
    background-color: #E9ECEF;
    box-shadow: none;
    overflow: hidden;
}
[data-testid="stHeader"]::before { display: none !important; }

html body [data-testid="stDecoration"],
body [data-testid="stDecoration"],
[data-testid="stDecoration"] {
    display: none !important;
    background: transparent !important;
    width: 0 !important;
    min-width: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
    z-index: -1 !important;
}

[data-testid="stSidebar"] { display: none; }

[data-testid="stAppViewContainer"] > section {
    padding-left: 1rem;
    padding-right: 1rem;
    padding-top: 80px;
}

/* --- Global responsive helpers --- */
.stMarkdown img, .stImage img, img {
    max-width: 100%;
    height: auto;
}

/* Reduce paddings on mobile */
@media (max-width: 768px) {
  [data-testid="stAppViewContainer"] > section {
    padding-left: 0.75rem;
    padding-right: 0.75rem;
    padding-top: 68px;
  }
}

/* --- Animations --- */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.content-card {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 25px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
    animation: fadeIn 0.5s ease-out;
}
.content-card h1, .content-card h2, .content-card h3 { color: #111; font-weight: 700; }

@keyframes gradient-animation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.visual-gallery img {
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.page-footer {
    text-align: center;
    color: #555;
    font-size: 16px;
    padding: 25px 0;
    line-height: 1.8;
}
.page-footer a { color: #7B4BFF; text-decoration: none; font-weight: 500; }
.page-footer i { font-size: 15px; }

[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 25px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    animation: fadeIn 0.5s ease-out;
}

.lottie-container { transition: transform 0.3s ease-out; }
.lottie-container:hover { transform: scale(1.05); }

div[data-testid="stButton"] > button {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    color: #333;
    font-weight: 600;
    font-size: 1.1rem;
    text-align: left;
    width: 100%;
    height: 140px;
    transition: all 0.3s ease;
}
div[data-testid="stButton"] > button p {
    font-size: 0.9rem; font-weight: 400; color: #444; margin-top: 5px;
}
div[data-testid="stButton"] > button:hover {
    background-color: #F9F9F9; color: #000;
    border: 1px solid #3498db;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    transform: translateY(-5px);
}

/* Predict submit button */
div[data-testid="stFormSubmitButton"] > button {
    background-color: #2ecc71; color: white; font-weight: bold; height: auto; text-align: center;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #27ae60; color: white; border: none; transform: translateY(0);
}

[data-testid="stSuccess"] { background-color: rgba(46, 204, 113, 0.1); border: 1px solid #2ecc71; color: #27ae60; font-size: 1.1rem; font-weight: 600; }
[data-testid="stWarning"] { background-color: #FFF3CD; border: 1px solid #FFECB5; color: #664D03; font-weight: 600; }
[data-testid="stExpander"] { background-color: #F9F9F9; border: 1px solid #E0E0E0; }
[data-testid="stInfo"] { background-color: rgba(52, 152, 219, 0.1); border-left: 5px solid #3498db; color: #2980b9; }
[data-testid="stBarChart"] text { fill: #333333 !important; }

/* Full-bleed nav container (if used elsewhere) */
.full-bleed-nav {
  position: relative;
  left: 50%;
  right: 50%;
  margin-left: -50vw;
  margin-right: -50vw;
  width: 100vw;
  background: #E9ECEF;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,.06);
  padding: 10px 12px;
  overflow: hidden;
}
.full-bleed-nav .nav-row {
  display: flex; align-items: center; gap: 8px;
  overflow-x: auto; white-space: nowrap; scrollbar-width: none;
}
.full-bleed-nav .nav-row::-webkit-scrollbar { display: none; }
.full-bleed-nav [data-testid="stHorizontalBlock"],
.full-bleed-nav [data-testid="stHorizontalBlock"] > div { background: transparent !important; }
.full-bleed-nav .container, .full-bleed-nav .nav { background: transparent !important; }

/* --- RESPONSIVE: Typography / paddings scaling --- */
@media (max-width: 1024px) {
  .content-card { padding: 20px; }
  .page-footer { font-size: 15px; }
}
@media (max-width: 768px) {
  div[data-testid="stButton"] > button { height: auto; padding: 16px; font-size: 1rem; }
  .page-footer { font-size: 14px; padding: 18px 0; }
}
@media (max-width: 600px) {
  .content-card { padding: 16px; }
  .page-footer { font-size: 13px; line-height: 1.7; }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------- MODEL LOADING  ----------------------
import zipfile
from io import BytesIO

DEMO_MODE = False

@st.cache_resource
def load_models_from_zip():
    """Downloads and loads ML model + encoders from Google Drive ZIP (Streamlit Secrets)."""
    try:
        zip_url = st.secrets["model"]["zip_url"]
        response = requests.get(zip_url)
        response.raise_for_status()

        # Extract model artifacts to a folder
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall("models")

        model = joblib.load("models/huntington_model_pipeline.pkl")
        feature_encoders = joblib.load("models/feature_encoders.pkl")
        target_encoder = joblib.load("models/target_encoder.pkl")

        with open("models/model_columns.json") as f:
            model_columns = json.load(f)

        return model, target_encoder, feature_encoders, model_columns

    except Exception as e:
        global DEMO_MODE
        DEMO_MODE = True
        st.warning("⚠️ Running in Demo Mode: model artifacts not found or failed to load.")
        st.error(f"Model loading error: {e}")
        return None, None, None, None


with st.spinner("⏳ Loading ML model... Please wait."):
    model, target_encoder, feature_encoders, model_columns = load_models_from_zip()

if not DEMO_MODE:
    st.success(" Model loaded successfully — predictions are LIVE!")
else:
    st.info("🧩 Using demo heuristic (educational mode only).")


def demo_predict_stage(row):
    motor = row.get('Motor_Score', 0)
    func = row.get('Functional_Capacity_Score', 100)
    cog = row.get('Cognitive_Score', 100)
    chorea = row.get('Chorea_Score', 0)
    if motor < 30 and func >= 70 and cog >= 70:
        return 'No Disease'
    if motor < 45 and func >= 60 and cog >= 60:
        return 'Early'
    if motor < 80 and func >= 30:
        return 'Middle'
    return 'Severe'

def set_page(page_name):
    st.session_state.nav_menu = page_name

PAGE_OPTIONS = ["Home", "About HD", "Stage Prediction Tool", "Resources", "Wellness & Support Tips"]
page_to_index = {page: i for i, page in enumerate(PAGE_OPTIONS)}

@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.error(f"Error loading Lottie animation: {e}")
        return None

# --- 1. Home ---
def show_home_page():
    import streamlit as st
    from streamlit_extras.add_vertical_space import add_vertical_space

    st.markdown(
        """
        <style>
        .info-text {
            background: linear-gradient(135deg, rgba(203,191,255,0.35), rgba(232,225,255,0.7));
            backdrop-filter: blur(8px);
            padding: 20px 26px;
            border-radius: 14px;
            font-size: 17px;
            line-height: 1.85;
            color: #2e2e2e;
            font-weight: 400;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            border: 1px solid rgba(180,160,255,0.4);
            transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
        }
        .info-text:hover {
            transform: translateY(-3px);
            background: linear-gradient(135deg, rgba(219,207,255,0.5), rgba(235,230,255,0.85));
            box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        }
        .info-text b { color: #6a3eff; font-weight: 600; }

        .section-title {
            font-size: 25px;
            font-weight: 750;
            color: #6238e8;
            margin-top: 30px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            position: relative;
            padding-bottom: 5px;
        }
        .section-title::after {
            content: "";
            position: absolute; bottom: 0; left: 0; width: 56px; height: 3px;
            background: linear-gradient(90deg, #7B4BFF, #bda9ff); border-radius: 2px;
        }
        .section-title span { margin-right: 10px; font-size: 1.2em; }

        [data-testid="stAppViewContainer"] > section { padding-top: 2rem; }

        .page-footer { text-align: center; color: #555; font-size: 14px; padding: 20px 0; line-height: 1.6; }
        .page-footer a { color: #7B4BFF; text-decoration: none; }

        /* --- HERO responsiveness --- */
        @keyframes gradientFlow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .hero-container {
            background: linear-gradient(-45deg, #c8bfff, #b8d0ff, #dcbfff, #b7a5ff);
            background-size: 600% 600%;
            animation: gradientFlow 6s ease infinite;
            border-radius: 20px;
            padding: 2rem 3rem;
            display: flex;
            align-items: center;
            justify-content: space-around;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            margin-bottom: 1rem;
            gap: 1.5rem;
        }
        .hero-text { max-width: 600px; text-align: left; }
        .hero-title { font-size: 2.6rem; font-weight: 700; color: #1b0c55; margin-bottom: 0.8rem; }
        .hero-subtitle { font-size: 1rem; color: #2d2d2d; line-height: 1.6; }
        .hero-image img {
            width: 230px; height: auto;
            animation: float 3s ease-in-out infinite, sway 5s ease-in-out infinite;
            filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.15));
            transition: transform 0.3s ease;
        }
        @keyframes float { 0% { transform: translateY(0); } 50% { transform: translateY(-14px);} 100% { transform: translateY(0);} }
        @keyframes sway { 0% { transform: translateX(0);} 50% { transform: translateX(10px);} 100% { transform: translateX(0);} }

        /* --- Responsive rules for Home --- */
        @media (max-width: 1024px) {
          .hero-title { font-size: 2.2rem; }
        }
        @media (max-width: 768px) {
          .hero-container { flex-direction: column; text-align: center; padding: 1.5rem; }
          .hero-text { max-width: 100%; text-align: center; }
          .hero-title { font-size: 1.5rem; }
          .hero-subtitle { font-size: 0.95rem; }
          .hero-image img { width: 180px; margin-top: 0.5rem; }
          .section-title { font-size: 20px; }
          .info-text { font-size: 15px; padding: 14px 18px; }
        }
        @media (max-width: 600px) {
          .hero-title { font-size: 1.7rem; }
          .hero-image img { width: 150px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    import base64
    from pathlib import Path

    image_path = BASE_DIR / "brain.png"

    if image_path.exists():
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
            img_src = f"data:image/png;base64,{encoded}"
    else:
        img_src = ""

    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-text">
                <div class="hero-title">HD Prognosis App</div>
                <div class="hero-subtitle">
                    An educational tool designed to help understand Huntington’s disease progression
                    and raise awareness through accessible, data-driven insights.
                </div>
            </div>
            <div class="hero-image">
                <img src="{img_src}" alt="Brain illustration"/>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-container">
            <div class="section-title"><span>👋</span>Welcome!</div>
            <div class="info-text">
                This app helps users explore and understand Huntington’s disease stages in an accessible way.
                It aims to raise awareness and empower families and individuals with information.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    add_vertical_space(1)

    st.markdown(
        """
        <div class="section-container">
            <div class="section-title"><span>🎯</span>Our Purpose</div>
            <div class="info-text">
                Understanding Huntington’s Disease can be overwhelming.
                Our goal is to make knowledge accessible, accurate, and empowering — for patients,
                families, and researchers alike. HD Predictor was developed as part of an initiative
                to bridge data science and healthcare education.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    add_vertical_space(1)

    st.markdown(
        """
        <div class="section-container">
            <div class="section-title"><span>⚙️</span>How HD Predictor Works</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="info-text">
                <b>🩺 Step 1 — Input Data:</b><br>
                Enter key clinical details such as motor, cognitive, and functional scores.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="info-text">
                <b>📊 Step 2 — Analyze:</b><br>
                The system analyzes your data using a machine learning model trained on clinical datasets.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="info-text">
                <b>🎯 Step 3 — Explore:</b><br>
                View the predicted stage and explore personalized educational resources.
            </div>
            """,
            unsafe_allow_html=True,
        )

    add_vertical_space(2)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button(
            "🧬 Learn about Huntington's Disease\n\nUnderstand the symptoms, stages, and how the condition is diagnosed.",
            on_click=set_page,
            args=("About HD",),
            use_container_width=True,
        )
    with col2:
        st.button(
            "📊 Stage Prediction Tool\n\nUse our tool to see a predicted stage based on clinical data.",
            on_click=set_page,
            args=("Stage Prediction Tool",),
            use_container_width=True,
        )
    with col3:
        st.button(
            "📚 Helpful Resources\n\nFind links to support groups, research, and tips for caregivers.",
            on_click=set_page,
            args=("Resources",),
            use_container_width=True,
        )

    add_vertical_space(2)

    st.markdown(
        """
        <div class="section-container">
            <div class="section-title"><span>🧠</span>Learn More</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.button(
            "🧬 What causes HD?\n\nLearn how genetic mutations lead to symptoms.",
            on_click=set_page,
            args=("About HD",),
            use_container_width=True,
        )
    with col2:
        st.button(
            "🧍 Understanding stages\n\nWhat changes happen as the disease progresses?",
            on_click=set_page,
            args=("About HD",),
            use_container_width=True,
        )
    with col3:
        st.button(
            "❤️ Caring for loved ones\n\nEmotional and lifestyle support tips.",
            on_click=set_page,
            args=("Wellness & Support Tips",),
            use_container_width=True,
        )
    with col4:
        st.button(
            "🔬 Current research\n\nPromising therapies and global initiatives.",
            on_click=set_page,
            args=("Resources",),
            use_container_width=True,
        )

    st.markdown("---", unsafe_allow_html=True)

    footer_html = """
    <div class="page-footer">
        <b>© 2025 HD Predictor</b><br>
        <i>
            This project is for educational purposes only and not a substitute for professional medical advice.<br>
            Always consult a qualified healthcare provider for diagnosis or treatment.
        </i>
        <br><br>
        Educational Tool |
        <a href="https://mail.google.com/mail/?view=cm&fs=1&to=varundube99@example.com&su=HD%20Predictor%20Inquiry&body=Hello%20Varun,%0D%0A%0D%0AI%20would%20like%20to%20ask%20about..."
           target="_blank" title="Email Varun Dubey">📩 Contact</a>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# --- 2. About HD ---
def show_about_hd_page():
    import streamlit as st
    from streamlit_extras.add_vertical_space import add_vertical_space
    from PIL import Image
    import base64
    from io import BytesIO

    def image_to_base64(img_path):
        img = Image.open(img_path)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    st.markdown("""
        <style>
            .hero-left {
                display: inline-flex !important;
                flex-direction: column !important;
                align-items: flex-start !important;
                text-align: left !important;
                width: 100% !important;
                padding: 1rem 3rem 1.5rem 3rem !important;
            }
            .hero-left h1 {
                color: #4F2D9D !important;
                font-size: 2.5rem !important;
                font-weight: 800 !important;
                margin-bottom: 0.6rem !important;
                display: flex !important;
                align-items: center !important;
                gap: 12px !important;
            }
            .hero-left p {
                color: #4B3C7A !important;
                font-size: 1.1rem !important;
                line-height: 1.6 !important;
                margin-top: 0.3rem !important;
                max-width: 800px !important;
            }

            /* Lightbox and modal */
            .img-container { position: relative; cursor: pointer; transition: transform 0.3s ease; }
            .img-container:hover { transform: scale(1.02); }
            .modal { display: none; position: fixed; z-index: 9999; left: 0; top: 0; width: 100%; height: 100%;
                     background-color: rgba(0,0,0,0.9); justify-content: center; align-items: center; }
            .modal img { max-width: 90%; max-height: 90%; border-radius: 12px; box-shadow: 0 0 20px rgba(255,255,255,0.2); }
            .modal:target { display: flex; }
            .close-btn { position: absolute; top: 30px; right: 50px; font-size: 2rem; color: white; text-decoration: none;
                         background: rgba(255,255,255,0.2); padding: 5px 14px; border-radius: 50%; transition: 0.2s; }
            .close-btn:hover { background: rgba(255,255,255,0.4); transform: scale(1.1); }

            .section-header {
                font-size: 1.5rem;
                font-weight: 800;
                color: #4b3db6;
                margin-top: 1.2rem;
                margin-bottom: 0.4rem;
                display: flex;
                align-items: center;
                font-family: 'Inter', sans-serif;
            }
            .section-header span.icon { font-size: 1.5rem; margin-right: 8px; }
            .section-box {
                background: linear-gradient(180deg, #ede8ff 0%, #e1d8ff 100%);
                border-radius: 14px;
                padding: 1.3rem 1.8rem;
                box-shadow: 0 4px 10px rgba(0,0,0,0.05);
                border-left: 5px solid #7B4BFF;
                font-size: 1.05rem;
                line-height: 1.7;
                color: #2b235a;
                margin-bottom: 1.5rem;
            }
            .section-box b { color: #3c2a8c; }
            .section-box:hover { transform: translateY(-2px); box-shadow: 0 6px 14px rgba(0,0,0,0.07); }

            /* Responsive for About page */
            @media (max-width: 1024px) {
              .hero-left { padding: 0.75rem 1.25rem 1.25rem 1.25rem !important; }
              .hero-left h1 { font-size: 2.2rem !important; }
            }
            @media (max-width: 768px) {
              .hero-left { padding: 0.5rem 1rem 1rem 1rem !important; }
              .hero-left h1 { font-size: 1.9rem !important; }
              .section-header { font-size: 1.25rem; }
              .section-box { font-size: 1rem; padding: 1rem 1.2rem; }
            }
            @media (max-width: 600px) {
              .hero-left h1 { font-size: 1.7rem !important; }
            }
        </style>

        <div class="hero-left">
            <h1>🧠 Understanding Huntington’s Disease</h1>
        </div>
    """, unsafe_allow_html=True)

    add_vertical_space(1)

    try:
        image1_base64 = image_to_base64(BASE_DIR / "HD1.png")
        image2_base64 = image_to_base64(BASE_DIR / "HD2.png")

    except Exception:
        st.warning("⚠️ Please ensure 'HD1.png' and 'HD2.png' are in the same directory.")
        return

    st.markdown("""
        <div class="section-header" style="margin-bottom:0.6rem;"><span class="icon">🧠</span>What is Huntington’s Disease?</div>
        <div class="section-box" style="margin-top:0.4rem;">
            Huntington’s disease (HD) is a <b>genetic brain disorder</b> that slowly affects the way a person moves, thinks, and feels. 
            It happens because of a <b>faulty gene</b> that causes certain nerve cells in the brain to stop working over time.  
            <br><br>
            People with HD may begin to experience <b>uncontrolled movements</b>, small changes in <b>mood or behavior</b>, 
            and trouble with <b>thinking or concentrating</b>. These changes usually start gradually and become more noticeable as years pass.
            <br><br>
            The condition is <b>inherited</b>, which means it is passed from parent to child. 
            If one parent has the faulty gene, there is a <b>50% chance</b> that each child may also inherit it.  
            <br><br>
            Although there is no cure yet, understanding HD early helps people plan, stay active, 
            and get medical and emotional support to live as comfortably and independently as possible.
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <a href="#img1">
            <div class="img-container">
                <img src="data:image/png;base64,{image1_base64}" 
                    alt="HD Genetic Diagram" 
                    style="
                        width:70%;
                        border-radius:12px;
                        box-shadow:0 4px 10px rgba(0,0,0,0.1);
                        margin: 0 auto 10px;
                        display:block;
                    ">
            </div>
        </a>

        <div id="img1" class="modal">
            <a href="#" class="close-btn">×</a>
            <img src="data:image/png;base64,{image1_base64}">
        </div>
    """, unsafe_allow_html=True)

    add_vertical_space(1)

    st.markdown(f"""
        <a href="#img2">
            <div class="img-container">
                <img src="data:image/png;base64,{image2_base64}" 
                    alt="Brain Comparison" 
                    style="
                        width:70%;
                        border-radius:12px;
                        box-shadow:0 4px 10px rgba(0,0,0,0.1);
                        margin: 0 auto 14px;
                        display:block;
                    ">
            </div>
        </a>

        <div id="img2" class="modal">
            <a href="#" class="close-btn">×</a>
            <img src="data:image/png;base64,{image2_base64}">
        </div>

        <div class="section-header"><span class="icon">🧬</span>Causes & Genetics</div>
        <div class="section-box">
            HD is caused by a single mutated gene, known as the <b>HTT gene</b>. 
            This gene contains a repeated section of DNA, called a <b>'CAG repeat'</b>. 
            In people with HD, this section is repeated too many times.
            <br><br>
            This genetic difference results in a toxic protein that damages brain cells. 
            Because the gene is <b>'autosomal dominant,'</b> a person only needs to inherit 
            one copy from one parent to develop the condition.
        </div>
    """, unsafe_allow_html=True)

    add_vertical_space(1)

    st.markdown("""
        <div class="section-header"><span class="icon">⚕️</span>Common Symptoms</div>
        <div class="section-box">
            Symptoms vary greatly from person to person but typically fall into three categories:
            <ul>
                <li><b>Motor Symptoms:</b> Involuntary jerking or twitching (chorea), problems with balance,
                    muscle rigidity, and difficulty with speech or swallowing.</li>
                <li><b>Cognitive Symptoms:</b> Difficulty organizing tasks, trouble focusing, memory lapses,
                    and impaired decision-making.</li>
                <li><b>Psychiatric Symptoms:</b> Depression, anxiety, irritability, mood swings, and social withdrawal.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    add_vertical_space(1)

    st.markdown("""
        <div class="section-header"><span class="icon">📈</span>Stages of Huntington’s Disease</div>
        <div class="section-box">
            The progression of HD is typically described in three stages, though the experience 
            is unique for every individual:
            <ol>
                <li><b>Early Stage:</b> Mild symptoms — slight coordination issues, subtle movements, 
                    or mood changes. Individuals can usually live independently.</li>
                <li><b>Middle Stage:</b> Movement and speech become harder. Chorea intensifies, 
                    and assistance with daily tasks is often needed.</li>
                <li><b>Late Stage:</b> Full-time care required. Major difficulties in motor control, 
                    speech, and swallowing. Focus shifts to comfort and support.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# --- 3. Prediction ---
def show_prediction_page():
    import streamlit as st
    import pandas as pd

    st.title("📊 Stage Prediction Tool")

    st.markdown("""
        <style>
        div[data-testid="stForm"] label,
        div[data-testid="stNumberInputLabel"],
        div[data-testid="stSelectboxLabel"],
        div[data-testid="stTextInputLabel"],
        div[data-baseweb="form-control"] label,
        div[data-testid="stMarkdownContainer"] p {
            color: #1F3B64 !important;
            font-weight: 600 !important;
            font-size: 0.96rem !important;
            letter-spacing: 0.3px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        .section-heading {
            background: linear-gradient(90deg, #eaf0ff 0%, #dfe8ff 100%);
            color: #003366;
            padding: 14px 20px;
            border-radius: 12px;
            font-size: 1.4rem;
            font-weight: 800;
            letter-spacing: 0.3px;
            border-left: 6px solid #4B90FF;
            box-shadow: 0 3px 10px rgba(0,0,0,0.06);
            margin-top: 30px;
            margin-bottom: 15px;
            font-family: 'Inter', sans-serif;
        }
        .content-box {
            background: linear-gradient(180deg, #ffffff 0%, #f6f9ff 100%);
            border: 1px solid #d7e3ff;
            border-radius: 16px;
            padding: 22px 26px;
            margin-bottom: 30px;
            font-size: 1.08rem;
            line-height: 1.8;
            color: #2a2a2a;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        .content-box:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08); }
        .content-box b { color: #004b91; font-weight: 700; }
        .content-box i { color: #333; font-style: italic; }
        .content-box::first-letter { font-size: 1.3rem; }
        hr { border: none; border-top: 1px solid #e6eef8; margin: 22px 0; }

        [data-testid="stInfo"] {
            background: linear-gradient(90deg, #edf5ff 0%, #e6f0ff 100%);
            border-left: 6px solid #4B90FF;
            color: #003b7a;
            font-size: 1rem;
            border-radius: 12px;
            padding: 15px 22px;
            margin-top: 20px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        }

        /* Form card + inputs */
        .content-card { background-color: #f9fcff; padding: 1.5rem; border-radius: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin-top: 1rem; transition: all 0.3s; }

        div[data-baseweb="input"] > div {
            background-color: #e8f4fa !important; 
            border: 1px solid #b5e0f0 !important;
            color: #000000 !important;
            border-radius: 10px !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #e8f4fa !important;
            border: 1px solid #b5e0f0 !important;
            color: #000000 !important;
            border-radius: 10px !important;
        }
        div[role="listbox"], div[data-baseweb="popover"], ul[role="listbox"] {
            background-color: #e8f4fa !important; color: #000000 !important; border: 1px solid #b5e0f0 !important; border-radius: 10px !important;
        }
        div[role="option"]:hover, li[role="option"]:hover { background-color: #d7edf8 !important; }

        div[data-baseweb="input"] button {
            background-color: #2f3640 !important; color: #ffffff !important; border: none !important; border-radius: 6px !important;
        }
        div[data-baseweb="input"] button:hover { background-color: #414b57 !important; }

        .section-heading { background-color: #e0f0ff; color: #004c91; padding: 10px 15px; border-radius: 8px; font-size: 22px; font-weight: 600; margin-top: 25px; margin-bottom: 10px; }
        .content-box { background-color: #f4f8fb; padding: 15px 20px; border-radius: 10px; font-size: 17px; color: #333; border: 1px solid #dbe7f2; margin-bottom: 20px; line-height: 1.6; }

        /* Responsive for Prediction page */
        @media (max-width: 1024px) {
          .section-heading { font-size: 20px; }
          .content-box { font-size: 16px; padding: 16px 18px; }
        }
        @media (max-width: 768px) {
          .section-heading { font-size: 18px; padding: 10px 12px; }
          .content-box { font-size: 15px; padding: 14px 16px; }
        }
        </style>
    """, unsafe_allow_html=True)

    if not model and not DEMO_MODE:
        st.error("Model is not loaded. Cannot proceed with prediction.")
        st.stop()

    with st.form("prediction_form"):
        st.markdown("""
            <style>
            .content-card { background-color: #f9fcff; padding: 1.5rem; border-radius: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin-top: 1rem; transition: all 0.3s ease-in-out; }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <style>
        div[data-baseweb="select"] > div {
            background-color: #e8f4fa !important;
            border: 1px solid #b5e0f0 !important;
            color: #000000 !important;
            border-radius: 10px !important;
        }
        div[role="listbox"], div[data-baseweb="popover"], ul[role="listbox"] {
            background-color: #e8f4fa !important; color: #000000 !important; border: 1px solid #b5e0f0 !important; border-radius: 10px !important;
        }
        div[role="option"]:hover, li[role="option"]:hover { background-color: #d7edf8 !important; }

        /* Inputs spacing on mobile */
        @media (max-width: 768px) {
          [data-testid="stNumberInput"], [data-testid="stSelectbox"] { margin-bottom: 8px; }
        }
        </style>
        """, unsafe_allow_html=True)

        st.header("Patient & Clinical Information")
        st.markdown("Enter the patient's details below. This information helps estimate the likely stage based on clinical patterns and research data.")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Current Age", min_value=1, max_value=120, value=65)
            sex = st.selectbox("Sex", ['Male', 'Female'])
            family_history = st.selectbox("Family History of Huntington's", ['Yes', 'No'])
            age_of_onset = st.number_input("Age of Symptom Onset", min_value=1, max_value=120, value=55)

        with col2:
            htt_cag_repeat = st.number_input("HTT CAG Repeat Length", min_value=10, max_value=100, value=45)
            motor_score = st.number_input("Motor Score", min_value=0, max_value=124, value=50)
            cognitive_score = st.number_input("Cognitive Score", min_value=0, max_value=100, value=40)
            chorea_score = st.number_input("Chorea Score", min_value=0.0, max_value=28.0, value=10.0, step=0.1)

        functional_score = st.number_input(
            "Functional Capacity Score (0-100)",
            min_value=0, max_value=100, value=35,
            help="A score from 0 (total dependence) to 100 (fully independent)."
        )

        submitted = st.form_submit_button("Predict Disease Stage")

    if submitted:
        sex_numeric = 1 if sex == 'Male' else 0
        family_history_numeric = 1 if family_history == 'Yes' else 0

        input_data = {
            'Age': age, 'Sex': sex_numeric, 'Family_History': family_history_numeric,
            'HTT_CAG_Repeat_Length': htt_cag_repeat, 'Age_of_Onset': age_of_onset,
            'Motor_Score': motor_score, 'Cognitive_Score': cognitive_score,
            'Chorea_Score': chorea_score, 'Functional_Capacity_Score': functional_score,
            'Gene/Factor': 'HTT', 'Function': 'CAG Trinonucleotide Repeat Expansion',
            'Effect': 'Neurodegeneration', 'Category': 'Primary Cause'
        }

        input_df = pd.DataFrame([input_data])
        input_df['Disease_Duration'] = input_df['Age'] - input_df['Age_of_Onset']
        input_df['Disease_Duration'] = input_df['Disease_Duration'].clip(lower=0)

        if feature_encoders:
            for col, encoder in feature_encoders.items():
                if col in input_df.columns and col not in ['Sex', 'Family_History']:
                    try:
                        input_df[col] = encoder.transform(input_df[col])
                    except Exception as e:
                        st.warning(f"Could not encode feature {col}: {e}")

        try:
            if model_columns:
                for col in model_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[model_columns]
            else:
                st.error("Model columns not loaded.")
                st.stop()
        except Exception as e:
            st.error(f"Unexpected error during data prep: {e}")
            st.stop()

        final_prediction = "No Disease"
        try:
            if DEMO_MODE or model is None or target_encoder is None:
                final_prediction = demo_predict_stage(input_data)
            else:
                prediction_encoded = model.predict(input_df)
                final_prediction = target_encoder.inverse_transform(prediction_encoded)[0]
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()
        
        import plotly.graph_objects as go

        def render_stage_gauge(stage):
            stage_levels = {"No Disease": 0.1, "Early": 0.4, "Middle": 0.7, "Severe": 1.0}
            value = stage_levels.get(stage, 0.1)
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge",
                value=value * 100,
                gauge={
                    "shape": "angular",
                    "axis": {
                        "range": [0, 100],
                        "tickmode": "array",
                        "tickvals": [12.5, 37.5, 62.5, 87.5],
                        "ticktext": ["No Disease", "Early", "Middle", "Severe"],
                        "tickfont": {"size": 15, "color": "#2a2a2a", "family": "Inter, sans-serif"},
                    },
                    "bar": {"color": "#003366", "thickness": 0.2},
                    "bgcolor": "#ffffff",
                    "steps": [
                        {"range": [0, 25], "color": "#a5f0b3"},
                        {"range": [25, 50], "color": "#f9e79f"},
                        {"range": [50, 75], "color": "#f8c471"},
                        {"range": [75, 100], "color": "#f1948a"},
                    ],
                    "threshold": {
                        "line": {"color": "#003366", "width": 6},
                        "thickness": 0.8,
                        "value": value * 100,
                    },
                },
                domain={'x': [0, 1], 'y': [0, 1]},
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=20, r=20, t=30, b=0),
                font={"color": "#2a2a2a", "size": 18},
            )
            fig.add_annotation(
                text=f"<b>{stage}</b>",
                x=0.5, y=0.1, showarrow=False,
                font={"color": "#003366", "size": 24, "family": "Inter, sans-serif"}
            )
            return fig

        st.markdown('<div class="section-heading">🧠 Stage Severity Indicator</div>', unsafe_allow_html=True)
        st.plotly_chart(render_stage_gauge(final_prediction), use_container_width=True)
        
        st.markdown('<div class="section-heading">Prediction Result</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="content-box">
                Based on the details provided, our analysis suggests the predicted stage is: 
                <b>{final_prediction}</b>.
                <br>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f'<div class="section-heading"> \'{final_prediction}\' Stage</div>', unsafe_allow_html=True)

        if final_prediction == 'Early':
            st.markdown(
                """
                <div class="content-box">
                    In the <b>early stage</b> of Huntington’s disease, changes may be mild and gradual.  
                    Subtle issues such as small movement difficulties, mild balance changes, mood shifts, or problems with focus and concentration may begin to appear.  
                    <br>
                    Most people can continue with work, hobbies, and daily responsibilities independently.  
                    Regular medical check-ups and early lifestyle adjustments can help slow down progression and improve quality of life.
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif final_prediction == 'Middle':
            st.markdown(
                """
                <div class="content-box">
                    The <b>middle stage</b> usually involves more noticeable symptoms.  
                    Movements can become slower or more rigid, and tasks like writing, speaking, or walking may require more effort.  
                    Cognitive and emotional changes — such as forgetfulness, frustration, or anxiety — may also become more apparent.  
                    <br>
                    At this stage, people often benefit from structured routines, supportive therapies, and occasional assistance with daily activities.
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif final_prediction == 'Severe':
            st.markdown(
                """
                <div class="content-box">
                    The <b>advanced stage</b> of Huntington’s disease is marked by significant loss of motor control and communication abilities.  
                    People may rely on full-time care for eating, movement, and personal hygiene.  
                    Cognitive awareness may still be present, so compassionate care and emotional support are especially important.  
                    <br>
                    Medical teams often focus on comfort, dignity, and symptom relief — ensuring the person’s environment is calm, safe, and nurturing.
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif final_prediction == 'No Disease':
            st.markdown(
                """
                <div class="content-box">
                    Your results do <b>not suggest active features</b> of Huntington’s disease based on typical clinical patterns.  
                    However, this does <b>not replace professional evaluation</b>.  
                    If you have a family history or ongoing neurological concerns, a neurologist or genetic counselor can help provide further testing and reassurance.  
                    <br>
                    Maintaining regular health check-ups and healthy lifestyle habits remains the best approach to long-term wellbeing.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-heading">Next Steps</div>', unsafe_allow_html=True)

        if final_prediction == 'Early':
            st.markdown(
                """
                <div class="content-box">
                    💡 Focus on early prevention and awareness.  
                    Maintain physical activity, eat a balanced diet, and stay socially connected.  
                    Discuss possible long-term care planning with healthcare professionals while independence is still high.  
                    <br>
                    Regular physiotherapy, speech therapy, and mindfulness-based activities may help preserve both mental and physical function.
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif final_prediction == 'Middle':
            st.markdown(
                """
                <div class="content-box">
                    💡 Prioritize daily safety and physical stability.  
                    Occupational and speech therapies can help adapt your home and communication for comfort and confidence.  
                    Emotional health is equally important — regular counseling and support groups can reduce stress and isolation.  
                    <br>
                    Family education at this stage can help prepare for care needs and strengthen support networks.
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif final_prediction == 'Severe':
            st.markdown(
                """
                <div class="content-box">
                    💡 Emphasis now shifts to <b>comfort and compassionate care</b>.  
                    Managing nutrition, preventing infections, and providing emotional reassurance are the primary goals.  
                    Specialized nursing, physiotherapy, and palliative care can greatly improve quality of life for both patients and caregivers.  
                    <br>
                    Caregivers are encouraged to seek community and respite support to maintain their own health and wellbeing.
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif final_prediction == 'No Disease':
            st.markdown(
                """
                <div class="content-box">
                    💡 Continue leading a healthy, active lifestyle and consider speaking to a healthcare professional for reassurance or screening.  
                    Genetic counseling can offer clarity if there is a family history of Huntington’s disease.  
                    Remember — early awareness and informed lifestyle choices can make a lasting difference.
                </div>
                """,
                unsafe_allow_html=True,
            )

# --- 4. Resources ---
def show_resources_page():
    st.title("📚 Helpful Resources")
    st.markdown("Here are some trustworthy, supportive links for learning support.")

    st.markdown("""
        <style>
        .main { background-color: #f4f6fb; }

        .section-header {
            background: linear-gradient(90deg, #d9ddff, #ececff);
            padding: 18px 28px;
            border-radius: 14px;
            margin-top: 25px;
            margin-bottom: 18px;
            font-weight: 700;
            font-size: 1.2rem;
            color: #1f1f3d;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
        }

        .resource-item {
            background: #ffffff;
            border-left: 5px solid #6c63ff;
            border-radius: 12px;
            padding: 15px 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .resource-item:hover {
            transform: translateY(-2px);
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        a { color: #4b60e5 !important; font-weight: 600; text-decoration: none; }
        a:hover { text-decoration: underline; }

        /* Responsive */
        @media (max-width: 768px) {
          .section-header { padding: 14px 16px; font-size: 1.05rem; }
          .resource-item { padding: 12px 14px; }
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🧭 Support & Information</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="resource-item">
            <p><a href="https://hdsa.org/" target="_blank">Huntington’s Disease Society of America (HDSA)</a><br>
            Dedicated to improving the lives of people with Huntington’s disease through research, support, and advocacy.</p>
        </div>

        <div class="resource-item">
            <p><a href="https://eurohuntington.org/" target="_blank">European Huntington Association (EHA)</a><br>
            Connects families and professionals across Europe to share knowledge, support, and initiatives.</p>
        </div>

        <div class="resource-item">
            <p><a href="https://en.hdbuzz.net/" target="_blank">HDBuzz</a><br>
            Research news in plain, easy-to-understand language for the global HD community.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📖 Educational Resources</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="resource-item">
            <p><a href="https://www.mayoclinic.org/diseases-conditions/huntingtons-disease/symptoms-causes/syc-20356117" target="_blank">Mayo Clinic</a><br>
            Reliable overview on symptoms, causes, and treatment options for Huntington’s disease.</p>
        </div>

        <div class="resource-item">
            <p><a href="https://www.nhs.uk/conditions/huntingtons-disease/" target="_blank">NHS (UK)</a><br>
            Trusted UK health guidance with information for patients and caregivers.</p>
        </div>

        <div class="resource-item">
            <p><a href="https://medlineplus.gov/huntingtonsdisease.html" target="_blank">MedlinePlus</a><br>
            Comprehensive medical library of information and latest research summaries.</p>
        </div>
    """, unsafe_allow_html=True)

# --- 5. Wellness & Support Tips ---
def show_wellness_page():
    import streamlit as st

    st.markdown("""
        <h1 style='text-align: center; 
                   background: -webkit-linear-gradient(45deg, #3a2e9a, #7a6ff0);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   font-size: 2.4rem;
                   font-weight: 800;
                   letter-spacing: 1px;
                   margin-bottom: 0;
                   '>🌿 Wellness & Support Tips</h1>
        <p style='text-align: center; color: #4e4f78; font-size: 1.1rem; margin-bottom: 35px;'>
            Practical, evidence-based recommendations to support mind and body wellness for patients.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        .content-card {
            background: linear-gradient(180deg, #e6e6ff 0%, #dcdcff 100%);
            border: 1px solid #b7b9ff;
            border-radius: 16px;
            padding: 30px 32px;
            margin: 30px 0;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }
        .content-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.12);
            background: linear-gradient(180deg, #ede9ff 0%, #d9d5ff 100%);
        }
        .content-title {
            color: #1f1d5c;
            font-size: 1.45rem;
            font-weight: 800;
            margin-bottom: 14px;
            display: flex; align-items: center;
        }
        .content-title span { font-size: 1.6rem; margin-right: 10px; }
        .content-card p { color: #22223b; line-height: 1.8; font-size: 1.07rem; margin-bottom: 10px; }
        .content-card ul { margin-left: 1.6rem; line-height: 1.9; color: #2f2f55; }
        .content-card li::marker { color: #4b46e0; }
        .highlight {
            background: linear-gradient(90deg, #dcf8ed 0%, #e7fff8 100%);
            padding: 20px; border-left: 6px solid #00a86b; border-radius: 12px;
            margin-top: 40px; color: #114b36; font-weight: 600; font-size: 1.05rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        }

        /* Responsive */
        @media (max-width: 1024px) {
          .content-card { padding: 24px; }
          .content-title { font-size: 1.3rem; }
        }
        @media (max-width: 768px) {
          .content-card { padding: 18px; }
          .content-title { font-size: 1.2rem; }
          .content-card p { font-size: 1rem; }
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card">
        <div class="content-title"><span>🏃‍♀️</span>Physical Activity & Brain Health</div>
        <p>Regular, physician-approved exercise is <b>strongly linked to improved coordination, balance, and emotional stability</b>.
        Gentle physical activity supports brain function, flexibility, and mood regulation.</p>
        <p><b>Try this:</b></p>
        <ul>
            <li>Aim for 20–30 minutes of low-impact exercise daily.</li>
            <li>Include balance and stretching activities like yoga or tai chi.</li>
            <li>Opt for joint-friendly routines such as swimming or stationary cycling.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card" style="background: linear-gradient(180deg, #e8f0ff 0%, #d7e2ff 100%); border-color: #a4b8ff;">
        <div class="content-title"><span>🥗</span>Nutrition for Neurological Wellness</div>
        <p>A well-balanced diet helps maintain focus, strength, and energy throughout the day. 
        Soft, nutrient-rich foods can make eating easier while supporting overall wellbeing.</p>
        <p><b>Try this:</b></p>
        <ul>
            <li>Choose easy-to-eat, nutrient-dense meals like soups, smoothies, and whole grains.</li>
            <li>Stay hydrated — even mild dehydration can impact concentration and mood.</li>
            <li>Consult a dietitian about supplements that may support brain metabolism.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="content-card" style="background: linear-gradient(180deg, #f0e8ff 0%, #e2d4ff 100%); border-color: #b09aff;">
        <div class="content-title"><span>🧘</span>Stress Management & Caregiver Support</div>
        <p>Emotional well-being is vital for both patients and caregivers. Managing stress helps maintain 
        mental clarity, improves mood, and strengthens resilience through daily routines.</p>
        <p><b>Try this:</b></p>
        <ul>
            <li>Practice 5 minutes of slow breathing or mindfulness daily.</li>
            <li>Keep a gratitude or reflection journal to release emotional tension.</li>
            <li>Caregivers: connect with local support groups to share experiences and prevent burnout.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight">
        🌸 <b>Remember:</b> Wellness is built one mindful choice at a time — a calm breath, a balanced meal, and a moment of gratitude can all help nurture better days.
    </div>
    """, unsafe_allow_html=True)

# --- 6. Top Navigation ---
if "nav_menu" not in st.session_state:
    st.session_state.nav_menu = "Home"

selected_page = option_menu(
    menu_title=None, 
    options=PAGE_OPTIONS, 
    icons=["house-fill", "info-circle-fill", "clipboard-data-fill", "book-half", "activity"], 
    menu_icon="cast", 
    default_index=page_to_index[st.session_state.nav_menu],
    orientation="horizontal",
    styles={
        "container": {
            "background-color": "#E9ECEF",
            "padding": "10px 0px",
            "margin-bottom": "20px",
            "border-radius": "0px",
            "box-shadow": "none",
            "position": "static",
            "width": "100vw",
            "margin-left": "calc(50% - 50vw)",
            "margin-right": "calc(50% - 50vw)",
            #Make the option menu scrollable on small screens 
            "overflow-x": "auto",
            "white-space": "nowrap"
        },
        "icon": {"color": "#666666", "font-size": "1.1rem"}, 
        "nav-link": {
            "font-size": "1.1rem", 
            "text-align": "center", 
            "margin":"0px 5px", 
            "padding": "10px 15px", 
            "color": "#333333",
            "border-radius": "5px",
            "transition": "all 0.3s ease"
        },
        "nav-link:hover": { "background-color": "#FFFFFF", "color": "#000000" },
        "nav-link-selected": {
            "background-color": "#FFFFFF", 
            "color": "#000000", 
            "font-weight": "600",
            "border-bottom": "3px solid #7B4BFF"
        },
    }
)

# Additional responsive tweak for the nav via inline style tag (font-size down on narrow)
st.markdown("""
<style>
/* Ensure top nav remains usable on small screens */
@media (max-width: 768px) {
  .container .nav-link, .container .nav-item a {
    font-size: 0.95rem !important;
    padding: 8px 10px !important;
  }
}
@media (max-width: 600px) {
  .container .nav-link, .container .nav-item a {
    font-size: 0.9rem !important;
    padding: 8px 8px !important;
  }
}
</style>
""", unsafe_allow_html=True)

if selected_page != st.session_state.nav_menu:
    st.session_state.nav_menu = selected_page
    st.rerun()

if st.session_state.nav_menu == "Home":
    show_home_page()
elif st.session_state.nav_menu == "About HD":
    show_about_hd_page()
elif st.session_state.nav_menu == "Stage Prediction Tool":
    show_prediction_page()
elif st.session_state.nav_menu == "Resources":
    show_resources_page()
elif st.session_state.nav_menu == "Wellness & Support Tips":
    show_wellness_page()
