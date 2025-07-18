import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import tempfile
import shutil
import matplotlib.pyplot as plt

# --- Sleek, Creative, Animated CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
    }
    #MainMenu, footer, header {visibility: hidden;}
    body {
        min-height: 100vh;
        background: linear-gradient(120deg, #e0e7ff 0%, #f8fafc 50%, #c7d2fe 100%);
        background-size: 200% 200%;
        animation: gradientMove 8s ease-in-out infinite;
    }
    @keyframes gradientMove {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .main {
        background: rgba(255,255,255,0.65);
        border-radius: 22px;
        box-shadow: 0 8px 40px rgba(37,99,235,0.13);
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        margin-top: 1.2rem;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        backdrop-filter: blur(8px) saturate(1.2);
        -webkit-backdrop-filter: blur(8px) saturate(1.2);
        animation: fadeIn 1.2s;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(30px);}
        to {opacity: 1; transform: none;}
    }
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #2563eb 100%);
        color: white;
        border-radius: 14px;
        font-weight: 600;
        padding: 0.7rem 0;
        font-size: 1.15rem;
        margin-top: 0.7rem;
        box-shadow: 0 4px 16px rgba(99,102,241,0.13);
        transition: background 0.2s, box-shadow 0.2s, transform 0.15s;
        border: none;
        width: 100%;
        min-width: 180px;
        max-width: 100%;
        display: block;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #6366f1 100%);
        box-shadow: 0 8px 32px rgba(99,102,241,0.18);
        transform: translateY(-2px) scale(1.03);
    }
    .stProgress > div > div > div > div {background-color: #6366f1;}
    .stFileUploader {background-color: #f1f5f9; border-radius: 12px;}
    .stTextInput>div>div>input {border-radius: 12px;}
    .stAlert {border-radius: 12px;}
    .st-bb {background-color: #f1f5f9; border-radius: 12px;}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {color: #2563eb;}
    .result-card {
        background: rgba(255,255,255,0.85);
        border-radius: 18px;
        box-shadow: 0 6px 32px rgba(99,102,241,0.13);
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        margin: 1.2rem 0;
        text-align: center;
        transition: box-shadow 0.2s, transform 0.18s;
        max-width: 500px;
        margin-left: auto;
        margin-right: auto;
        backdrop-filter: blur(4px) saturate(1.1);
        -webkit-backdrop-filter: blur(4px) saturate(1.1);
        animation: fadeIn 1.2s;
    }
    .result-card:hover {
        box-shadow: 0 12px 48px rgba(99,102,241,0.22);
        transform: scale(1.025);
    }
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 1rem;
        margin-top: 2.5rem;
        padding-bottom: 1.2rem;
        letter-spacing: 0.01em;
    }
    .custom-section {
        background: rgba(255,255,255,0.75);
        border-radius: 22px;
        box-shadow: 0 2px 12px rgba(99,102,241,0.09);
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        margin-bottom: 1.5rem;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        backdrop-filter: blur(6px) saturate(1.1);
        -webkit-backdrop-filter: blur(6px) saturate(1.1);
        animation: fadeIn 1.2s;
    }
    .custom-label {
        color: #2563eb;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .custom-caption {
        color: #64748b;
        font-size: 0.98rem;
        margin-bottom: 0.5rem;
    }
    .section-divider {
        border: none;
        border-top: 2px solid #6366f1;
        margin: 2.2rem 0 1.2rem 0;
    }
    .sticky-tabs {
        position: sticky;
        top: 0;
        z-index: 100;
        background: rgba(255,255,255,0.85);
        box-shadow: 0 2px 12px rgba(99,102,241,0.07);
        border-radius: 0 0 18px 18px;
        margin-bottom: 1.2rem;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        animation: fadeIn 1.2s;
    }
    .fab {
        position: fixed;
        bottom: 32px;
        right: 32px;
        background: linear-gradient(90deg, #6366f1 0%, #2563eb 100%);
        color: #fff;
        border: none;
        border-radius: 50%;
        width: 64px;
        height: 64px;
        box-shadow: 0 8px 32px rgba(99,102,241,0.18);
        font-size: 2.2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        z-index: 9999;
        transition: background 0.2s, box-shadow 0.2s, transform 0.15s;
    }
    .fab:hover {
        background: linear-gradient(90deg, #2563eb 0%, #6366f1 100%);
        box-shadow: 0 16px 48px rgba(99,102,241,0.28);
        transform: scale(1.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.markdown(
    """
    <div style='display: flex; align-items: center; justify-content: center; gap: 1rem;'>
        <h1 style='color:#2563eb; margin-bottom:0; letter-spacing:0.01em;'>Cats vs Dogs Classifier</h1>
    </div>
    <p style='text-align:center; color:#475569; font-size:19px; margin-bottom:1.5rem;'>A professional tool for image classification using deep learning.</p>
    """,
    unsafe_allow_html=True
)

MODEL_PATH = "cats_vs_dogs_mobilenetv2.h5"

@st.cache_resource
def load_pretrained_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img, target_size=(160, 160)):
    img = img.convert("RGB").resize(target_size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_with_model(model, img):
    arr = preprocess_image(img)
    pred = model.predict(arr)[0][0]
    label = "Dog" if pred > 0.5 else "Cat"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, float(confidence)

# --- Sticky Navigation Tabs ---
st.markdown('<div class="sticky-tabs"></div>', unsafe_allow_html=True)
tabs = st.tabs(["Pretrained Model", "Train Your Own Model", "About"])

# --- Pretrained Model Tab ---
with tabs[0]:
    st.markdown("<div class='custom-section'>", unsafe_allow_html=True)
    st.header("Use Pretrained Model")
    st.write("Upload an image of a cat or dog and get an instant prediction!")
    col1, col2 = st.columns([2, 3], gap="large")
    with col1:
        uploaded_file = st.file_uploader(
            "Upload an image to classify (Cat or Dog)",
            type=["jpg", "jpeg", "png"],
            key="pretrained_upload"
        )
    with col2:
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            with st.spinner("Classifying..."):
                model = load_pretrained_model()
                label, confidence = predict_with_model(model, img)
            st.markdown(f"""
            <div class='result-card' style='border-left: 6px solid #2563eb;'>
                <h3 style='color:#2563eb;'>Prediction: {label}</h3>
                <p style='font-size:18px;'>Confidence: <b>{confidence*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
    st.subheader("Example Training Curves (Pretrained Model)")
    st.caption("These are example curves for illustration only. The actual training history is not available for the pretrained model.")
    example_epochs = np.arange(1, 11)
    example_train_acc = [0.65, 0.72, 0.78, 0.82, 0.86, 0.89, 0.91, 0.93, 0.94, 0.95]
    example_val_acc =   [0.63, 0.70, 0.75, 0.80, 0.83, 0.86, 0.88, 0.89, 0.90, 0.91]
    example_train_loss = [0.65, 0.55, 0.45, 0.38, 0.32, 0.28, 0.24, 0.21, 0.19, 0.17]
    example_val_loss =   [0.68, 0.60, 0.52, 0.46, 0.41, 0.37, 0.34, 0.32, 0.30, 0.29]
    col3, col4 = st.columns(2)
    with col3:
        fig_acc, ax_acc = plt.subplots(figsize=(4.5,3.5))
        ax_acc.plot(example_epochs, example_train_acc, marker='o', label='Train Accuracy', color='#2563eb')
        ax_acc.plot(example_epochs, example_val_acc, marker='s', label='Validation Accuracy', color='#f59e42')
        ax_acc.set_xlabel("Epoch", fontsize=11)
        ax_acc.set_ylabel("Accuracy", fontsize=11)
        ax_acc.set_title("Accuracy over Epochs", fontsize=13)
        ax_acc.legend()
        ax_acc.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_acc)
    with col4:
        fig_loss, ax_loss = plt.subplots(figsize=(4.5,3.5))
        ax_loss.plot(example_epochs, example_train_loss, marker='o', label='Train Loss', color='#2563eb')
        ax_loss.plot(example_epochs, example_val_loss, marker='s', label='Validation Loss', color='#f59e42')
        ax_loss.set_xlabel("Epoch", fontsize=11)
        ax_loss.set_ylabel("Loss", fontsize=11)
        ax_loss.set_title("Loss over Epochs", fontsize=13)
        ax_loss.legend()
        ax_loss.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_loss)

# --- Train Your Own Model Tab ---
with tabs[1]:
    st.markdown("<div class='custom-section'>", unsafe_allow_html=True)
    st.header("Train Your Own Model")
    st.write("""
    <span style='color:#475569;'>
    Enter two class names, upload images for each, and train a custom classifier. You can use any categories you like!
    </span>
    """, unsafe_allow_html=True)
    up1, up2 = st.columns(2, gap="large")
    with up1:
        class1_name = st.text_input("Enter name for Class 1", value="Class A", key="class1_name")
        class1_files = st.file_uploader(
            f"Upload images for {class1_name}",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="class1"
        )
        if class1_files:
            st.caption(f"{len(class1_files)} images uploaded for {class1_name}")
            st.image([Image.open(f) for f in class1_files[:3]], width=100, caption=[f"{class1_name} {i+1}" for i in range(min(3, len(class1_files)))])
    with up2:
        class2_name = st.text_input("Enter name for Class 2", value="Class B", key="class2_name")
        class2_files = st.file_uploader(
            f"Upload images for {class2_name}",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="class2"
        )
        if class2_files:
            st.caption(f"{len(class2_files)} images uploaded for {class2_name}")
            st.image([Image.open(f) for f in class2_files[:3]], width=100, caption=[f"{class2_name} {i+1}" for i in range(min(3, len(class2_files)))])
    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
    train_col, _ = st.columns([2, 3])
    with train_col:
        train_clicked = st.button("Train Model", use_container_width=True)
    if train_clicked:
        if not class1_files or not class2_files:
            st.markdown("""
            <div class='result-card' style='border-left: 6px solid #ef4444;'>
                <h4 style='color:#ef4444;'>Please upload images for both classes.</h4>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"Training model to classify **{class1_name}** vs **{class2_name}**. This may take a moment...")
            progress = st.progress(0, text="Starting training...")
            tmpdir = tempfile.mkdtemp()
            try:
                class1_dir = os.path.join(tmpdir, class1_name)
                class2_dir = os.path.join(tmpdir, class2_name)
                os.makedirs(class1_dir, exist_ok=True)
                os.makedirs(class2_dir, exist_ok=True)
                for i, file in enumerate(class1_files):
                    img = Image.open(file).convert("RGB")
                    img.save(os.path.join(class1_dir, f"{class1_name}_{i}.jpg"))
                for i, file in enumerate(class2_files):
                    img = Image.open(file).convert("RGB")
                    img.save(os.path.join(class2_dir, f"{class2_name}_{i}.jpg"))
                from tensorflow.keras.preprocessing.image import ImageDataGenerator
                datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
                train_gen = datagen.flow_from_directory(tmpdir, target_size=(64,64), batch_size=8, class_mode='binary', subset='training')
                val_gen = datagen.flow_from_directory(tmpdir, target_size=(64,64), batch_size=8, class_mode='binary', subset='validation')
                class_indices = train_gen.class_indices
                # Reverse mapping: index -> class name
                index_to_class = {v: k for k, v in class_indices.items()}
                # Simple CNN model
                model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                epochs = 5
                history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
                for epoch in range(epochs):
                    hist = model.fit(train_gen, validation_data=val_gen, epochs=1, verbose=0)
                    # Collect metrics
                    history['accuracy'].append(hist.history['accuracy'][0])
                    history['val_accuracy'].append(hist.history['val_accuracy'][0])
                    history['loss'].append(hist.history['loss'][0])
                    history['val_loss'].append(hist.history['val_loss'][0])
                    progress.progress((epoch+1)/epochs, text=f"Training... Epoch {epoch+1}/{epochs}")
                st.markdown("""
                <div class='result-card' style='border-left: 6px solid #22c55e;'>
                    <h3 style='color:#22c55e;'>Model trained!</h3>
                </div>
                """, unsafe_allow_html=True)
                st.session_state['custom_model'] = model
                st.session_state['custom_class_names'] = (class1_name, class2_name)
                st.session_state['index_to_class'] = index_to_class
                # Save model and offer download
                model.save("custom_cnn_model.h5")
                with open("custom_cnn_model.h5", "rb") as f:
                    st.download_button("Download Model", f, file_name="custom_cnn_model.h5")
                st.success("Training Complete!")
                st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
                st.subheader("Training & Validation Accuracy")
                fig_acc, ax_acc = plt.subplots(figsize=(5.5,3.5))
                ax_acc.plot(history['accuracy'], marker='o', label='Train Accuracy', color='#2563eb')
                ax_acc.plot(history['val_accuracy'], marker='s', label='Validation Accuracy', color='#f59e42')
                ax_acc.set_xlabel("Epoch", fontsize=11)
                ax_acc.set_ylabel("Accuracy", fontsize=11)
                ax_acc.set_title("Accuracy over Epochs", fontsize=13)
                ax_acc.legend()
                ax_acc.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig_acc)
                st.subheader("Training & Validation Loss")
                fig_loss, ax_loss = plt.subplots(figsize=(5.5,3.5))
                ax_loss.plot(history['loss'], marker='o', label='Train Loss', color='#2563eb')
                ax_loss.plot(history['val_loss'], marker='s', label='Validation Loss', color='#f59e42')
                ax_loss.set_xlabel("Epoch", fontsize=11)
                ax_loss.set_ylabel("Loss", fontsize=11)
                ax_loss.set_title("Loss over Epochs", fontsize=13)
                ax_loss.legend()
                ax_loss.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig_loss)
            finally:
                shutil.rmtree(tmpdir)
    st.markdown("</div>", unsafe_allow_html=True)
    if 'custom_model' in st.session_state and 'custom_class_names' in st.session_state and 'index_to_class' in st.session_state:
        st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
        st.markdown("<div class='custom-section'>", unsafe_allow_html=True)
        st.subheader("Test Your Model")
        st.write("Upload a test image to see the prediction with your custom-trained model.")
        test_file = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"], key="test_custom")
        if test_file:
            img = Image.open(test_file).convert("RGB")
            st.image(img, caption="Test Image", use_container_width=True)
            arr = np.array(img.resize((64,64))) / 255.0
            arr = np.expand_dims(arr, axis=0)
            model = st.session_state['custom_model']
            index_to_class = st.session_state['index_to_class']
            pred = model.predict(arr)[0][0]
            # Binary classification: 0 or 1
            pred_class_idx = int(pred > 0.5)
            label = index_to_class[pred_class_idx]
            confidence = pred if pred > 0.5 else 1 - pred
            st.markdown(f"""
            <div class='result-card' style='border-left: 6px solid #2563eb;'>
                <h3 style='color:#2563eb;'>Prediction: {label}</h3>
                <p style='font-size:18px;'>Confidence: <b>{confidence*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- About Tab ---
with tabs[2]:
    st.markdown("<div class='custom-section'>", unsafe_allow_html=True)
    st.header("About This Project")
    st.write("""
    **Cats vs Dogs Classifier** is a professional web tool for image classification using deep learning. It demonstrates:
    - Use of transfer learning (MobileNetV2) for fast, accurate predictions
    - Custom model training on your own images
    - Modern, business-friendly UI with Streamlit
    
    **Developed by:** Your Name Here  
    **Powered by:** Streamlit, TensorFlow, Keras
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class='footer'>
    <hr style='margin-bottom:10px;'/>
    <span>Cats vs Dogs Classifier &copy; 2024 | <a href='https://streamlit.io/' target='_blank' style='color:#2563eb;'>Streamlit</a></span>
</div>
""", unsafe_allow_html=True) 