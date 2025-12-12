import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import shutil

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Fruit Detection App 🍎🍌🍊",
    page_icon="🍓",
    layout="centered"
)

# -------------------- TITLE --------------------
st.markdown("""
<h1 style='text-align: center; color: #ff4b4b;'>🍎🍌🍊Fruit Object Detection using YOLOv8</h1>
<p style='text-align: center;'>Upload an image and detect Apples, Bananas & Oranges</p>
<hr>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return YOLO("best_fruit_model.pt")

model = load_model()

# Function to clear old YOLO prediction folders
def clear_old_predictions():
    runs_dir = "runs/detect"
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)   # delete all previous predictions

# -------------------- IMAGE UPLOAD --------------------
uploaded_file = st.file_uploader("📤 Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # Show uploaded image
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Image")


    # Detect button
    if st.button("🔍 Detect Fruits"):

        with st.spinner("Detecting..."):

            # 1️⃣ Clear all old outputs
            clear_old_predictions()

            # 2️⃣ Save new temp image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                image.save(temp.name)
                temp_path = temp.name

            # 3️⃣ Run YOLO detection
            results = model.predict(source=temp_path, conf=0.25, save=True)

            # 4️⃣ Get only the latest prediction folder
            output_dir = results[0].save_dir

            # 5️⃣ Find the prediction image
            pred_image = None
            for f in os.listdir(output_dir):
                if f.endswith(".jpg"):
                    pred_image = os.path.join(output_dir, f)
                    break

            # 6️⃣ Display only the new detected image
            if pred_image:
                st.image(pred_image, caption="Detected Output")

        st.success("Detection Complete!")

 # -------------------- FOOTER --------------------
st.markdown("""
---
👨‍🎓 **Project:** Fruit Object Detection  
🧠 **Model:** YOLOv8  
🍎 **Classes:** Apple, Banana, Orange  
By **Anubriya.B**
""")