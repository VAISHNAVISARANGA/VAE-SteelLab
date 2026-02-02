import streamlit as st
import torch
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Import the logic from your generated files
from cnn_vae_steel import Config, load_trained_model, predict_from_image

# Page configuration
st.set_page_config(page_title="Steel Microstructure Analyzer", layout="wide")

st.title("🔬 Steel Microstructure Property Predictor")
st.markdown("""
This application uses a **CNN-VAE** model to analyze microstructure images and predict 
heat treatment parameters, phase fractions, and mechanical properties.
""")

# 1. Load Model and Scaler
@st.cache_resource
def init_model():
    model_path = "cnn_vae_steel.pth"
    scaler_path = "target_scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Model or Scaler files not found! Please ensure 'cnn_vae_steel.pth' and 'target_scaler.pkl' are in the app directory.")
        return None, None
        
    model, scaler = load_trained_model(model_path, scaler_path)
    return model, scaler

model, scaler = init_model()

# 2. Sidebar - Upload Image
st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload a Microstructure Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Microstructure")
        st.image(img, use_container_width=True)
        
    with col2:
        if st.button("Analyze Microstructure"):
            with st.spinner("Running CNN-VAE Inference..."):
                # Save temp file for the inference script logic
                temp_path = "temp_inference_img.png"
                img.save(temp_path)
                
                # Perform Prediction
                results = predict_from_image(temp_path, model, scaler, show_image=False)
                
                # Cleanup temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                # --- Display Results ---
                st.success("Analysis Complete!")
                
                # Category 1: Mechanical Properties
                st.subheader("🛠️ Mechanical Properties")
                mech_cols = st.columns(4)
                mech_data = results['Mechanical Properties']
                mech_cols[0].metric("Hardness", f"{mech_data['Hardness_HRC']:.1f} HRC")
                mech_cols[1].metric("Yield Strength", f"{mech_data['Yield_Strength']:.0f} MPa")
                mech_cols[2].metric("Tensile Strength", f"{mech_data['Tensile_Strength']:.0f} MPa")
                mech_cols[3].metric("Elongation", f"{mech_data['Elongation']:.1f} %")

                st.divider()

                # Category 2: Microstructural Fractions
                st.subheader("🧬 Microstructural Composition")
                micro_cols = st.columns(2)
                
                # Table/List View
                phase_data = results['Microstructure']
                df_phases = pd.DataFrame({
                    "Phase": ["Martensite", "Bainite", "Ferrite", "Pearlite"],
                    "Fraction (%)": [
                        phase_data['Martensite_Pct'], 
                        phase_data['Bainite_Pct'], 
                        phase_data['Ferrite_Pct'], 
                        phase_data['Pearlite_Pct']
                    ]
                })
                micro_cols[0].table(df_phases)
                micro_cols[1].metric("Grain Size", f"{phase_data['Grain_Size']:.1f} μm")

                st.divider()

                # Category 3: Process Parameters
                st.subheader("🔥 Predicted Heat Treatment Parameters")
                proc_data = results['Process Parameters']
                p_col1, p_col2, p_col3 = st.columns(3)
                
                p_col1.write(f"**Austenitizing:** {proc_data['Aust_Temp']:.1f}°C for {proc_data['Aust_Time']:.0f} min")
                p_col2.write(f"**Cooling Rate:** {proc_data['Cooling_Rate']:.1f} °C/s")
                p_col3.write(f"**Tempering:** {proc_data['Temp_Temp']:.1f}°C for {proc_data['Temp_Time']:.0f} min")

else:
    st.info("Please upload a microstructure image in the sidebar to begin analysis.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Steel Analysis System v1.0 | CNN-VAE Architecture")