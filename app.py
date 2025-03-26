import streamlit as st
import joblib
import numpy as np
from streamlit_extras.colored_header import colored_header
from streamlit_extras.let_it_rain import rain

# Load the trained model
model = joblib.load("xgboost_best_model.pkl")

# Streamlit UI Configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="💉",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .st-emotion-cache-1y4p8pa {
        padding: 2rem 1rem 10rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .risk-high {
        color: #ff4b4b;
        font-size: 1.5rem;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        background-color: #ffecec;
    }
    .risk-low {
        color: #06d6a0;
        font-size: 1.5rem;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        background-color: #e8f9f3;
    }
    .input-label {
        font-weight: 600;
        color: #2c3e50;
    }
    .footer {
        font-size: 0.8rem;
        text-align: center;
        margin-top: 2rem;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with enhanced UI
with st.sidebar:
    st.markdown("# 💉 Diabetes Predictor WebApp")
    st.markdown("""
    ### 🔍 How it works
    This advanced AI tool predicts diabetes risk using:
    - XGBoost Classifier model
    - 8 key health indicators
    - Clinical accuracy
    
    ### 📊 Model Performance
    - Accuracy: 95.50%
    - Precision: 93.91%
    - Recall: 92.29%
    - F1 Score: 93.09%
    - AUC Score: 99.20%
    - Log Loss: 0.1086
    
    ⚠️ **Note:** This tool doesn't replace professional medical advice.
    """)
    
    # Add a fun emoji animation when sidebar opens
    if st.sidebar.button("🎉 Celebrate Health"):
        rain(
            emoji="🎈",
            font_size=20,
            falling_speed=5,
            animation_length=1,
        )

# Main content area
colored_header(
    label="🔍 Diabetes Risk Assessment Tool",
    description="Enter patient details to predict diabetes likelihood",
    color_name="blue-70",
)

st.markdown("### 📝 Patient Health Profile")

# Organize inputs in columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="input-label">🤰 Pregnancies</p>', unsafe_allow_html=True)
    pregnancies = st.number_input("", min_value=0, max_value=20, step=1, key="preg", label_visibility="collapsed")
    
    st.markdown('<p class="input-label">💉 Plasma Glucose (mg/dL)</p>', unsafe_allow_html=True)
    plasma_glucose = st.number_input("", min_value=0, max_value=300, step=1, key="glucose", label_visibility="collapsed")
    
    st.markdown('<p class="input-label">❤️ Diastolic BP (mm Hg)</p>', unsafe_allow_html=True)
    diastolic_bp = st.number_input("", min_value=0, max_value=200, step=1, key="bp", label_visibility="collapsed")
    
    st.markdown('<p class="input-label">📏 Triceps Thickness (mm)</p>', unsafe_allow_html=True)
    triceps_thickness = st.number_input("", min_value=0, max_value=100, step=1, key="triceps", label_visibility="collapsed")

with col2:
    st.markdown('<p class="input-label">🩸 Serum Insulin (μU/mL)</p>', unsafe_allow_html=True)
    serum_insulin = st.number_input("", min_value=0, max_value=1000, step=1, key="insulin", label_visibility="collapsed")
    
    st.markdown('<p class="input-label">⚖️ BMI (kg/m²)</p>', unsafe_allow_html=True)
    bmi = st.number_input("", min_value=0.0, max_value=100.0, step=0.1, key="bmi", label_visibility="collapsed")
    
    st.markdown('<p class="input-label">🧬 Diabetes Pedigree</p>', unsafe_allow_html=True)
    diabetes_pedigree = st.number_input("", min_value=0.0, max_value=3.0, step=0.01, key="pedigree", label_visibility="collapsed")
    
    st.markdown('<p class="input-label">👨‍⚕️ Age (years)</p>', unsafe_allow_html=True)
    age = st.number_input("", min_value=0, max_value=120, step=1, key="age", label_visibility="collapsed")

# Predict button with enhanced UI
if st.button("🚀 Predict Diabetes Risk", use_container_width=True):
    # Prepare input data
    input_data = np.array([[pregnancies, plasma_glucose, diastolic_bp, triceps_thickness,
                            serum_insulin, bmi, diabetes_pedigree, age]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100  # Convert to percentage
    
    # Display results with enhanced UI
    st.markdown("## 📊 Prediction Results")
    
    if prediction == 1:
        rain(
            emoji="⚠️",
            font_size=20,
            falling_speed=3,
            animation_length=0.5,
        )
        st.markdown(f'<p class="risk-high">🚨 High Diabetes Risk: {probability:.2f}% probability</p>', unsafe_allow_html=True)
        st.warning("Recommendation: Please consult with a healthcare professional for further evaluation and management.")
    else:
        rain(
            emoji="✅",
            font_size=20,
            falling_speed=3,
            animation_length=0.5,
        )
        st.markdown(f'<p class="risk-low">✅ Low Diabetes Risk: {100 - probability:.2f}% confidence</p>', unsafe_allow_html=True)
        st.success("Maintain healthy lifestyle habits and regular check-ups to prevent diabetes.")
    
    # Add a visual meter
    st.markdown(f"""
    ### 📈 Risk Meter
    """)
    st.progress(int(probability))
    st.caption(f"Risk Score: {probability:.1f}/100")

# Additional information section
st.markdown("---")
st.markdown("""
## ℹ️ About Diabetes
Diabetes is a chronic condition that affects how your body turns food into energy. 
Early detection and management can prevent complications like:
- Heart disease
- Kidney damage
- Vision problems
- Nerve damage
""")

# Footer with social links
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Developed with ❤️ by Ramdhan | 
    📧 <a href="mailto:prajapatramdhan2001@gmail.com">prajapatramdhan2001@gmail.com</a> |  
    💻 <a href="https://github.com/ramdhan9604" target="_blank">GitHub</a> | 
    💼 <a href="https://linkedin.com/in/ramdhan9604" target="_blank">LinkedIn</a></p>
    <p>© 2023 Diabetes Predictor Pro. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)