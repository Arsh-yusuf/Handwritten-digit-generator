# Handwritten-digit-generator

This is a web app that generates images of handwritten digits (0–9) using a GAN trained from scratch.

---

## 🚀 Project Overview

This project implements a **Generative Adversarial Network (GAN)** trained on the MNIST dataset to generate 28x28 grayscale images of digits.

- Users can select a digit (0–9)
- The app generates **5 synthetic images** of that digit
- The model is trained from scratch using **PyTorch**
- The web app is built with **Streamlit**

---

## 📎 Live App Link

🔗 **[Launch the app here](https://tcu5thdd2ebbe8uybcfnme.streamlit.app/)**  

---

## 🏗️ Files in This Repository

| File           | Description                              |
|----------------|------------------------------------------|
| `app.py`       | Streamlit app code for digit generation  |
| `generator.py` | The Generator model architecture         |
| `generator.pth`| Trained model weights (PyTorch state_dict) |
| `requirements.txt` | Python dependencies for deployment   |
| `HandwrittenNumberGAN.ipynb` | Colab notebook used for training |

---

## 🧪 Model Training Details

- Dataset: **MNIST**
- Model: Custom **DCGAN-style Generator**
- Trained on: **Google Colab with T4 GPU**
- Framework: **PyTorch**
- No pre-trained weights were used

---

## 📦 Installation (For Local Testing)

```bash
pip install -r requirements.txt
streamlit run app.py
