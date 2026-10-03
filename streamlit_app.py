import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Suppress TensorFlow verbose logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import keras
    from keras import layers
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers

# Page Configuration
st.set_page_config(
    page_title="Footwear Style & Shopping Hub",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .info-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .pred-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-high {
        background-color: #DCFCE7;
        color: #15803D;
    }
    .badge-med {
        background-color: #FEF3C7;
        color: #B45309;
    }
    .badge-low {
        background-color: #F1F5F9;
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)

# Directory & Model Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_PATH = os.path.join(BASE_DIR, "footwear_classifier_model.keras")
H5_PATH = os.path.join(BASE_DIR, "footwear_classifier_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "class_labels.json")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

CATEGORY_ICONS = {
    "boots": "",
    "flip_flops": "",
    "loafers": "",
    "sandals": "",
    "sneakers": "",
    "soccer_shoes": ""
}

CATEGORY_NAMES = {
    "boots": "Boots",
    "flip_flops": "Flip Flops",
    "loafers": "Loafers",
    "sandals": "Sandals",
    "sneakers": "Sneakers",
    "soccer_shoes": "Soccer Shoes"
}

# Model Reconstruction for Backward Compatibility
def build_mobilenet_classifier(num_classes=6):
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    base_model.trainable = False
    model = keras.Sequential([
        keras.Input(shape=(224, 224, 3)),
        base_model,
        layers.GlobalAveragePooling2D(name="global_average_pooling2d"),
        layers.Dropout(0.3, name="dropout"),
        layers.Dense(128, activation="relu", name="dense"),
        layers.Dense(num_classes, activation="softmax", name="dense_1")
    ])
    return model

@st.cache_resource(show_spinner="Initializing AI classifier...")
def load_footwear_model():
    """Load model with automatic fallback and conversion."""
    if not os.path.exists(LABELS_PATH):
        st.error(f"Missing label file at: {LABELS_PATH}")
        st.stop()
        
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    # 1. Native Keras 3 format
    if os.path.exists(KERAS_PATH):
        try:
            model = keras.models.load_model(KERAS_PATH)
            return model, labels
        except Exception:
            pass

    # 2. Rebuild & load weights from HDF5
    if os.path.exists(H5_PATH):
        try:
            model = build_mobilenet_classifier(num_classes=len(labels))
            model.load_weights(H5_PATH)
            model.save(KERAS_PATH)
            return model, labels
        except Exception as e:
            st.error(f"Failed to initialize model from weights: {e}")
            st.stop()

    st.error("No model file found (.keras or .h5).")
    st.stop()

model, class_labels = load_footwear_model()

# Footwear Knowledge Base
FOOTWEAR_KNOWLEDGE_BASE = {
    "boots": {
        "tagline": "Durable, rugged, and protective footwear for all-weather confidence.",
        "style_tips": "Pair with raw or dark indigo denim, tailored corduroys, or flannel shirts. For a sophisticated modern look, combine Chelsea boots with slim trousers and an overcoat.",
        "care_guide": "Wipe dirt off after wear. Treat smooth leather with wax conditioner every 3 months. For suede or nubuck, use a crepe brush and hydrophobic silicone spray.",
        "occasion": "Outdoor adventures, winter seasons, casual workdays, rainy weather.",
        "price_range": "₹2,500 – ₹14,000",
        "stores": ["Timberland", "Red Tape", "Woodland", "Clarks", "Myntra", "Amazon"]
    },
    "flip_flops": {
        "tagline": "Ultralight, breathable, open-toe leisure footwear for warm escapes.",
        "style_tips": "Ideal with swim trunks, linen beach shorts, boardies, or lightweight relaxed home loungewear.",
        "care_guide": "Rinse clean with fresh water immediately after chlorine or seawater exposure. Avoid leaving under direct burning sunlight for prolonged periods.",
        "occasion": "Poolside, beach days, locker rooms, indoor lounging.",
        "price_range": "₹300 – ₹2,500",
        "stores": ["Havaianas", "Puma", "Nike", "Decathlon", "Flipkart", "Amazon"]
    },
    "loafers": {
        "tagline": "Timeless laceless slip-ons balancing relaxed elegance and office polish.",
        "style_tips": "Flawlessly complements ankle-grazing chinos, tapered formal trousers, linen blazers, and polo shirts. Best worn sockless or with invisible no-show socks.",
        "care_guide": "Insert cedar shoe trees immediately after wearing to retain contour. Buff regularly with matching cream polish; brush suede variants gently.",
        "occasion": "Business casual, dinner dates, weddings, rooftop evenings.",
        "price_range": "₹1,800 – ₹10,000",
        "stores": ["Bata", "Louis Philippe", "Hush Puppies", "AJIO", "Myntra"]
    },
    "sandals": {
        "tagline": "Ergonomic, open-strap comfort engineered for warm-weather adventures.",
        "style_tips": "Pair with breezy linen trousers, cargo shorts, or casual summer vacation fits. Strappy leather sandals also look effortless with relaxed button-down shirts.",
        "care_guide": "Spot clean straps with a gentle soapy sponge. If cork-based (like Birkenstocks), periodically apply cork sealer to prevent cracking.",
        "occasion": "Summer travels, casual weekend errands, tropical vacations.",
        "price_range": "₹800 – ₹5,500",
        "stores": ["Birkenstock", "Crocs", "Bata", "Red Chief", "Amazon", "Myntra"]
    },
    "sneakers": {
        "tagline": "The ultimate street, athleisure, and everyday lifestyle staple.",
        "style_tips": "Limitless versatility: rock low-tops with ankle cuffs or joggers, high-tops with relaxed cargo pants, and minimal white clean sneakers with slim suits.",
        "care_guide": "Wipe rubber outsoles with a specialized sneaker cleaner or magic eraser. Use sneaker crease shields and avoid high-heat dryer cycles.",
        "occasion": "Daily casual wear, college, city travel, parties, fitness.",
        "price_range": "₹2,000 – ₹18,000+",
        "stores": ["Nike", "Adidas", "Puma", "Vans", "VegNonVeg", "Superkicks", "Myntra"]
    },
    "soccer_shoes": {
        "tagline": "High-traction cleated footwear engineered for pitch performance and agility.",
        "style_tips": "Pair with cushioned sports grip socks, team kits, athletic track pants, or breathable performance shorts designed for active training.",
        "care_guide": "Knock off dried mud and turf directly after match play. Wipe synthetic uppers clean with a damp cloth; stuff with paper to dry in shaded ventilation.",
        "occasion": "Football pitch, turf matches, agility training, sports practice.",
        "price_range": "₹1,500 – ₹12,000",
        "stores": ["Nike Football", "Adidas Performance", "Puma Football", "Decathlon"]
    }
}

def get_shopping_links(category):
    query = category.replace('_', '+')
    return {
        "Amazon": f"https://www.amazon.in/s?k=buy+{query}+shoes",
        "Myntra": f"https://www.myntra.com/{category.replace('_', '-')}",
        "AJIO": f"https://www.ajio.com/search/?text={query}",
        "Flipkart": f"https://www.flipkart.com/search?q={query}+shoes",
        "Google Shopping": f"https://www.google.com/search?tbm=shop&q=buy+{query}+shoes+online"
    }

# Sidebar Content
with st.sidebar:
    st.title(" Footwear Advisor")
    st.caption("AI-Powered Footwear Recognition & Style Guide")
    
    st.markdown("---")
    st.markdown("###  Model Diagnostics")
    st.success("Model Status: **Ready & Online**")
    st.info(f"**Backbone:** MobileNetV2\n\n**Categories:** {len(class_labels)} Classes\n\n**Input Resolution:** 224 x 224 px")
    
    st.markdown("---")
    st.markdown("### ️ Supported Categories")
    for key in class_labels:
        icon = CATEGORY_ICONS.get(key, "")
        name = CATEGORY_NAMES.get(key, key.capitalize())
        st.write(f"{icon} **{name}**")
        
    st.markdown("---")
    st.caption("Developed with TensorFlow/Keras & Streamlit.")

# Main Application Header
st.markdown('<div class="main-header"> Footwear Style & Shopping Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a photo, snap an image with your camera, or pick a sample to identify footwear styles, care tips, and top shopping links.</div>', unsafe_allow_html=True)

# Input Methods Tabs
tab_upload, tab_camera, tab_samples = st.tabs([" Upload Image", " Use Camera", " Try Sample"])

selected_image = None
image_source_label = ""

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a footwear photo...",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a crisp, well-lit image of the footwear"
    )
    if uploaded_file is not None:
        try:
            selected_image = Image.open(uploaded_file).convert("RGB")
            image_source_label = f"Uploaded File: {uploaded_file.name}"
        except Exception as err:
            st.error(f"Error reading image: {err}")

with tab_camera:
    camera_photo = st.camera_input("Take a snapshot of any footwear...")
    if camera_photo is not None:
        try:
            selected_image = Image.open(camera_photo).convert("RGB")
            image_source_label = "Camera Snapshot"
        except Exception as err:
            st.error(f"Error reading camera photo: {err}")

with tab_samples:
    st.write("Don't have an image ready? Select one of the test samples below:")
    sample_files = []
    if os.path.exists(SAMPLES_DIR):
        sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if sample_files:
        chosen_sample = st.selectbox("Select sample:", sample_files)
        if st.button("Load Selected Sample"):
            sample_path = os.path.join(SAMPLES_DIR, chosen_sample)
            selected_image = Image.open(sample_path).convert("RGB")
            image_source_label = f"Sample: {chosen_sample}"
    else:
        st.info("No sample files found in samples/ directory.")

# Prediction and Display Flow
if selected_image is not None:
    st.markdown("---")
    col_img, col_pred = st.columns([1, 1.2], gap="large")
    
    with col_img:
        st.subheader("️ Input Image")
        st.image(selected_image, use_container_width=True, caption=image_source_label)
        st.caption(f"Original Dimensions: {selected_image.width} × {selected_image.height} pixels")
        
    with col_pred:
        # Preprocess & Predict
        with st.spinner("Analyzing style and features..."):
            img_resized = selected_image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array, verbose=0)[0]
            top_idx = int(np.argmax(predictions))
            predicted_class = class_labels[top_idx]
            confidence = float(predictions[top_idx]) * 100.0
            
            icon = CATEGORY_ICONS.get(predicted_class, "")
            display_name = CATEGORY_NAMES.get(predicted_class, predicted_class.capitalize())

        # Top Prediction Banner
        st.subheader(" Detection Result")
        
        badge_class = "badge-high" if confidence >= 70 else ("badge-med" if confidence >= 40 else "badge-low")
        badge_text = "High Confidence" if confidence >= 70 else ("Moderate Confidence" if confidence >= 40 else "Low Confidence")
        
        st.markdown(f"""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); margin-bottom: 1rem;">
            <div style="font-size: 1.8rem; font-weight: 700; color: #0F172A;">
                {icon} {display_name}
            </div>
            <div style="margin-top: 0.5rem; display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.4rem; font-weight: 600; color: #2563EB;">{confidence:.1f}% Match</span>
                <span class="pred-badge {badge_class}">{badge_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Probabilities Breakdown
        st.markdown("**Probability Breakdown:**")
        chart_data = pd.DataFrame({
            "Style": [CATEGORY_NAMES.get(label, label.capitalize()) for label in class_labels],
            "Probability (%)": [float(p) * 100 for p in predictions]
        }).sort_values(by="Probability (%)", ascending=False)
        
        st.bar_chart(chart_data.set_index("Style"), height=220)

    # Style and Shopping Hub Section
    st.markdown("---")
    st.subheader(f" Footwear Knowledge & Shopping Hub: {icon} {display_name}")
    
    info = FOOTWEAR_KNOWLEDGE_BASE.get(predicted_class, {})
    
    if info:
        st.write(f"*{info.get('tagline', '')}*")
        
        col_style, col_care = st.columns(2, gap="medium")
        
        with col_style:
            st.markdown(f"""
            <div class="info-card">
                <h4 style="margin-top: 0; color: #1E293B;"> Styling Advice</h4>
                <p>{info.get('style_tips', 'N/A')}</p>
                <h4 style="color: #1E293B;"> Best Occasions</h4>
                <p>{info.get('occasion', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_care:
            st.markdown(f"""
            <div class="info-card">
                <h4 style="margin-top: 0; color: #1E293B;"> Care & Maintenance</h4>
                <p>{info.get('care_guide', 'N/A')}</p>
                <h4 style="color: #1E293B;"> Price Range</h4>
                <p>{info.get('price_range', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### ️ Where to Buy Online")
        st.write(f"**Top Brands & Stores:** {', '.join(info.get('stores', []))}")
        
        links = get_shopping_links(predicted_class)
        cols = st.columns(len(links))
        for idx, (platform, url) in enumerate(links.items()):
            with cols[idx]:
                st.link_button(f"Shop on {platform}", url, use_container_width=True)
    else:
        st.info("Additional styling data for this category will be available soon.")
else:
    st.info(" Please upload an image, use your camera, or pick a sample above to get started.")