import streamlit as st
import torch
from PIL import Image
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
import torch.nn as nn
import pandas as pd

# Define class names
class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']

@st.cache_resource
def load_model(model_path: str, num_classes: int):
    # Load pretrained architecture
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    # Replace the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=num_classes),
        nn.LogSoftmax(dim=1)
    )
    # Load your trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# Instantiate once and cache
model = load_model('models/trained_mobilenet.pt', len(class_names))
transform = get_transform()

st.title('Fruit Image Classifier')
st.write("""
🍎🍌🍇🥭🍓 **Welcome to the Fruit Classifier App!**

Snap a photo or upload an image, and this app will tell you which fruit it is — choosing from:

- **Apple**
- **Banana**
- **Grapes**
- **Mango**
- **Strawberry**

Let's get classifying! 🍽️✨
""")

input_method = st.radio("Choose image input method:", ("Upload from Gallery", "Take a Photo"))
if input_method == "Upload from Gallery":
    uploaded_image = st.file_uploader("Upload an image", type=("jpg", "jpeg", "png"))
else:
    uploaded_image = st.camera_input("Take a photo")

if uploaded_image:
    image = Image.open(uploaded_image).convert('RGB')
    col1, col2 = st.columns(2)
    with col1:
        st.image(image.resize((256, 256)), caption="Input Image")
    with col2:
        if st.button("Predict"):
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                logits = model(img_tensor)
                # Convert log-probabilities to probabilities
                probs = torch.exp(logits)
                # Predicted class index and confidence
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_idx].item() * 100  # in percentage
            
            st.success(f"🔍 Prediction: **{class_names[pred_idx]}**")
            st.info(f"📊 Confidence: **{confidence:.2f}%**")
            st.snow()
            
            # Prepare data for bar chart of all class probabilities
            probs_np = probs.squeeze().tolist()
            df = pd.DataFrame({
                'Fruit': class_names,
                'Confidence (%)': [p * 100 for p in probs_np]
            })
            st.bar_chart(df.set_index('Fruit'))
