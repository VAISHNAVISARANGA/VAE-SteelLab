
# Steel Microstructure Analysis & Property Prediction

This project implements a **Convolutional Variational Autoencoder (CNN-VAE)** designed for materials informatics. It bridges the gap between visual microstructure data and quantitative material properties, allowing users to predict heat treatment history, phase composition, and mechanical performance from a single micrograph.

---

## 📖 Reference Paper & Dataset Generation

The dataset used in this project consists of **500 samples** that were synthetically generated to replicate the physical laws and methodology described in the following research:

**Tiwari, S., Dash, K., Heo, S., Park, N., & Reddy, N. G. S. (2026).**
*"Machine Learning-Driven Prediction of Microstructural Evolution and Mechanical Properties in Heat-Treated Steels Using Gradient Boosting."*
**Crystals, 16(1), 61.**
DOI: 10.3390/cryst16010061

---

## ⚙️ Phenomenological Physics Engine

Because experimental metallurgical data is often proprietary or scarce, the 500-sample dataset was created using a synthetic engine based on:

* **Phase Transformations:** Martensite fractions calculated via Koistinen–Marburger (KM) kinetics:
  [ f_m = 1 - \exp[-k(M_s - T_q)] ]

* **Hardenability:** Phase evolution (Bainite, Ferrite, Pearlite) derived from chemistry and CCT (Continuous Cooling Transformation) logic.

* **Tempering Kinetics:** Hardness reduction modeled using the Hollomon-Jaffe Parameter (HJP).

* **Strengthening Mechanisms:** Yield and Tensile strengths incorporating the Hall-Petch Relationship, linking strength to the prior austenite grain size.

---

## 🧠 Core Model: CNN-VAE

The backbone of this project is a **Convolutional Variational Autoencoder combined with a Multi-Task Regression Head**. Unlike standard CNNs, the VAE learns the underlying distribution of the microstructure.

### Architecture Overview

* **Encoder:** Compresses the (128 \times 128) micrograph into a Latent Vector ((z)) of size 128.
* **Decoder:** Reconstructs the original image from the latent vector to ensure visual features like grain boundary density and phase distribution are captured.
* **Regression Head:** Simultaneously maps the latent vector (z) to **15 numerical properties** across three categories.

---

## 📊 Predicted Outputs

### Process Parameters

* (T_\gamma), (t_\gamma)
* Cooling Rate
* (T_q)
* (T_t), (t_t)

### Microstructural Outputs

* Martensite fraction
* Bainite fraction
* Ferrite fraction
* Pearlite fraction
* Grain Size

### Mechanical Properties

* Hardness (HRC)
* Yield Strength
* Tensile Strength
* Elongation

---

## 📂 Project Structure

```plaintext
├── cnn_vae_steel.py       # Model architecture and training logic
├── inference.py           # Script for single-image property prediction
├── app.py                 # Streamlit web application dashboard
├── requirements.txt       # Dependencies (torch, streamlit, etc.)
├── steel_heat_treatment_data.csv  # Synthetic dataset (500 samples)
└── microstructures/       # Generated micrographs
```

---

## 🚀 Installation & Setup

### 1. Environment

Ensure you have **Python 3.9+** installed. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

---

### 2. Training

To train the model on the synthetic dataset:

```bash
python cnn_vae_steel.py
```

This will generate:

* `cnn_vae_steel.pth` (trained weights)
* `target_scaler.pkl` (normalization parameters)

---

### 3. Running the Dashboard

Launch the interactive Streamlit interface:

```bash
streamlit run app.py
```

---

## 🧪 How to Use

1. Open the Streamlit URL (usually `http://localhost:8501`).
2. Upload a microstructure image via the sidebar.
3. Click **"Analyze Microstructure"**.
4. The system will output the predicted heat treatment parameters and mechanical performance metrics.
