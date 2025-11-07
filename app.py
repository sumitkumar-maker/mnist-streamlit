import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

# Neural Network Definition
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load Saved Model
model = NeuralNet()
model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Streamlit UI
st.title("🧠 MNIST Digit Classifier")
st.write("Upload a handwritten digit image (0–9)")

uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L")
    img = TF.invert(img)
    st.image(img, caption="Uploaded Image", width=200)

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img_tensor)
        _, pred = torch.max(out, 1)

    st.success(f"✅ Predicted Digit: {pred.item()}")
