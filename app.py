import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="AutoCare Centre & Engine Diagnostics",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 36px; font-weight: 800; color: #1E293B; margin-bottom: 5px; }
    .sub-header { font-size: 16px; color: #64748B; margin-bottom: 30px; }
    .card { background-color: #F8FAFC; padding: 22px; border-radius: 10px; border: 1px solid #E2E8F0; margin-bottom: 15px; }
    .metric-title { font-size: 14px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; display: block; margin-bottom: 15px; }
    .status-container { padding: 25px; border-radius: 12px; font-weight: 700; margin-top: 10px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
    .status-normal { background-color: #ECFDF5; border-left: 8px solid #10B981; color: #065F46; }
    .status-fault { background-color: #FEF2F2; border-left: 8px solid #EF4444; color: #991B1B; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_diagnostic_pipeline():
    """Loads the compiled Scikit-learn classification pipeline"""
    try:
        return joblib.load('models/engine_condition_pipeline.pkl')
    except FileNotFoundError:
        return None

pipeline = load_diagnostic_pipeline()

with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/car-service.png", width=75)
    st.markdown("## AutoCare Analytics")
    st.markdown("This dashboard feeds raw sensor data directly into the diagnostic engine to run automated vehicle state classifications.")
    st.markdown("---")
    
    st.markdown("### Model Engine Status")
    if pipeline is not None:
        st.success("● Inference Engine Active")
        model_architecture = type(pipeline.named_steps['model']).__name__
        st.info(f"Deployed Backend:\n`{model_architecture}`")
    else:
        st.error("❌ Inference Engine Offline")
        st.caption("Ensure your trained model pipeline object is compiled and saved under `models/engine_condition_pipeline.pkl`.")

st.markdown('<div class="main-header">⚙️ Engine Diagnostic Control Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time Telemetry Processing & Failure Risk Assessments</div>', unsafe_allow_html=True)

if pipeline is None:
    st.warning("⚠️ **Pipeline Binary Missing:** Streamlit cannot handle live inputs until the model training pipeline has run and exported the `models/engine_condition_pipeline.pkl` asset file.")
else:
    with st.form("live_sensor_stream_form"):
        st.markdown("### 📊 Live Telemetry Stream Channels")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<span class="metric-title">⚙️ Mechanical & Hydraulic Pressures</span>', unsafe_allow_html=True)
            engine_rpm = st.number_input("Engine Speed (RPM)", min_value=100, max_value=6000, value=800, step=50)
            lub_oil_press = st.number_input("Lubricating Oil Pressure (bar)", min_value=0.0, max_value=15.0, value=3.2, step=0.1)
            fuel_press = st.number_input("Fuel System Pressure (bar)", min_value=0.0, max_value=30.0, value=6.2, step=0.1)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<span class="metric-title">🌡️ Thermal Loops & Fluids</span>', unsafe_allow_html=True)
            coolant_press = st.number_input("Coolant Loop Pressure (bar)", min_value=0.0, max_value=12.0, value=2.3, step=0.1)
            lub_oil_temp = st.number_input("Lubricating Oil Temperature (°C)", min_value=10.0, max_value=180.0, value=78.0, step=0.5)
            coolant_temp = st.number_input("Engine Coolant Temperature (°C)", min_value=10.0, max_value=180.0, value=78.8, step=0.5)
            st.markdown('</div>', unsafe_allow_html=True)

        submit_diagnostic_scan = st.form_submit_button("🚀 Run Live Engine Diagnostics Scan", use_container_width=True)

    if submit_diagnostic_scan:
        input_data = pd.DataFrame([{
            'Engine rpm': engine_rpm,
            'Lub oil pressure': lub_oil_press,
            'Fuel pressure': fuel_press,
            'Coolant pressure': coolant_press,
            'lub oil temp': lub_oil_temp,
            'Coolant temp': coolant_temp
        }])

        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        scan_confidence = probabilities[prediction] * 100

        st.markdown("### 🎯 Scanner Assessment Report Summary")
        
        if prediction == 1:
            st.markdown(f"""
                <div class="status-container status-normal">
                    <h3 style="margin:0 0 5px 0; color:#065F46;">✓ CERTIFIED OPERATIONAL STATE</h3>
                    <p style="margin:0; font-size:15px; font-weight:400;">
                        No telemetry anomalies or critical parameter deviations detected. The asset is operating within stable baselines.
                        <br><b>Classification Certainty: {scan_confidence:.2f}%</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="status-container status-fault">
                    <h3 style="margin:0 0 5px 0; color:#991B1B;">🚨 FAULT WARNING SIGNATURE DETECTED</h3>
                    <p style="margin:0; font-size:15px; font-weight:400;">
                        Telemetry signals indicate a severe operational fault profile. Recommend mechanical review of underlying fluid lines or pressure pump efficiency.
                        <br><b>Classification Certainty: {scan_confidence:.2f}%</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)
