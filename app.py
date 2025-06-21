import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from generator import Generator
import numpy as np

# Setup
noise_dim = 64
device = torch.device("cpu")

# Load model
G = Generator(noise_dim).to(device)
G.load_state_dict(torch.load("generator.pth", map_location=device))
G.eval()

# Streamlit UI
st.title("🧠 Handwritten Digit Generator")
digit = st.selectbox("Pick a digit (0-9)", list(range(10)))
if st.button("Generate"):
    z = torch.randn(5, noise_dim).to(device)
    labels = torch.full((5,), digit, dtype=torch.long).to(device)
    labels_onehot = F.one_hot(labels, num_classes=10).float().unsqueeze(2).unsqueeze(3)
    z = z.view(5, noise_dim, 1, 1)
    input_vec = z  # No conditioning used; your GAN is unconditional

    with torch.no_grad():
        generated = G(input_vec).cpu() * 0.5 + 0.5

    st.write(f"Generated images for digit: {digit}")
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, img in enumerate(generated):
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)
