import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = Image.open(test_image).resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Disease classes
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Treatment suggestions for all diseases
treatment_info = {
    "Apple___Apple_scab": {
        "Organic": "Neem oil spray, sulfur-based fungicides, potassium bicarbonate",
        "Inorganic": "Captan, Mancozeb, Myclobutanil"
    },
    "Apple___Black_rot": {
        "Organic": "Prune infected areas, copper-based sprays, compost tea",
        "Inorganic": "Thiophanate-methyl, Ziram, Propiconazole"
    },
    "Apple___Cedar_apple_rust": {
        "Organic": "Sulfur sprays, remove nearby cedar trees, resistant varieties",
        "Inorganic": "Myclobutanil, Propiconazole, Mancozeb"
    },
    "Apple___healthy": {
        "Organic": "Regular pruning, balanced fertilization, compost application",
        "Inorganic": "Preventive fungicides, insecticides as needed"
    },
    "Blueberry___healthy": {
        "Organic": "Acidic mulch, proper spacing, balanced watering",
        "Inorganic": "Acidic fertilizers, preventive fungicides"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "Organic": "Potassium bicarbonate, neem oil, silica-based sprays",
        "Inorganic": "Trifloxystrobin, Myclobutanil, Propiconazole"
    },
    "Cherry_(including_sour)___healthy": {
        "Organic": "Proper pruning, balanced nutrition, foliar seaweed spray",
        "Inorganic": "Balanced NPK fertilizers, preventive sprays"
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "Organic": "Crop rotation, resistant varieties, compost tea sprays",
        "Inorganic": "Azoxystrobin, Pyraclostrobin, Chlorothalonil"
    },
    "Corn_(maize)___Common_rust_": {
        "Organic": "Resistant varieties, proper spacing, neem oil sprays",
        "Inorganic": "Azoxystrobin, Propiconazole, Tebuconazole"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "Organic": "Crop rotation, foliar compost tea, resistant varieties",
        "Inorganic": "Mancozeb, Pyraclostrobin, Propiconazole"
    },
    "Corn_(maize)___healthy": {
        "Organic": "Balanced organic fertilization, adequate irrigation, mulching",
        "Inorganic": "Balanced NPK fertilizers, micronutrient supplements"
    },
    "Grape___Black_rot": {
        "Organic": "Copper-based fungicides, proper pruning, adequate spacing",
        "Inorganic": "Myclobutanil, Mancozeb, Tebuconazole"
    },
    "Grape___Esca_(Black_Measles)": {
        "Organic": "Pruning wound treatments, balanced nutrition, biofungicides",
        "Inorganic": "Sodium arsenite (restricted), Trichoderma-based products"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "Organic": "Copper hydroxide, potassium bicarbonate, proper canopy management",
        "Inorganic": "Mancozeb, Azoxystrobin, Pyraclostrobin"
    },
    "Grape___healthy": {
        "Organic": "Balanced organic fertilization, proper pruning, adequate spacing",
        "Inorganic": "Balanced NPK fertilizers, preventive fungicides"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "Organic": "Reflective mulch, beneficial insects, nutritional therapy with zinc and manganese",
        "Inorganic": "Imidacloprid, Thiamethoxam (for vector control), enhanced nutrition program"
    },
    "Peach___Bacterial_spot": {
        "Organic": "Copper sprays, crop rotation, resistant varieties",
        "Inorganic": "Copper hydroxide, Oxytetracycline, Streptomycin (where allowed)"
    },
    "Peach___healthy": {
        "Organic": "Proper pruning, balanced fertilization, neem oil as preventive",
        "Inorganic": "Balanced NPK fertilizers, preventive fungicides"
    },
    "Pepper,_bell___Bacterial_spot": {
        "Organic": "Copper-based sprays, compost tea, crop rotation",
        "Inorganic": "Copper hydroxide, Mancozeb, Acibenzolar-S-methyl"
    },
    "Pepper,_bell___healthy": {
        "Organic": "Balanced organic fertilization, adequate calcium, proper spacing",
        "Inorganic": "Balanced NPK fertilizers with calcium supplements"
    },
    "Potato___Early_blight": {
        "Organic": "Compost tea, copper fungicides, crop rotation",
        "Inorganic": "Chlorothalonil, Mancozeb, Azoxystrobin"
    },
    "Potato___Late_blight": {
        "Organic": "Fixed copper sprays, resistant varieties, proper spacing",
        "Inorganic": "Metalaxyl, Dimethomorph, Fluopicolide"
    },
    "Potato___healthy": {
        "Organic": "Balanced fertilization, adequate irrigation, mulching",
        "Inorganic": "Balanced NPK fertilizers, preventive fungicides"
    },
    "Raspberry___healthy": {
        "Organic": "Proper pruning, balanced nutrition, compost application",
        "Inorganic": "Balanced NPK fertilizers, preventive fungicides"
    },
    "Soybean___healthy": {
        "Organic": "Crop rotation, inoculants for nitrogen fixation, balanced nutrients",
        "Inorganic": "Balanced NPK fertilizers, preventive fungicides"
    },
    "Squash___Powdery_mildew": {
        "Organic": "Potassium bicarbonate, neem oil, milk spray (1:10 dilution)",
        "Inorganic": "Sulfur compounds, Myclobutanil, Trifloxystrobin"
    },
    "Strawberry___Leaf_scorch": {
        "Organic": "Copper fungicides, proper spacing, adequate irrigation",
        "Inorganic": "Captan, Thiophanate-methyl, Azoxystrobin"
    },
    "Strawberry___healthy": {
        "Organic": "Balanced nutrition, adequate drainage, straw mulch",
        "Inorganic": "Balanced NPK fertilizers, preventive fungicides"
    },
    "Tomato___Bacterial_spot": {
        "Organic": "Copper sprays, bio-control agents, crop rotation",
        "Inorganic": "Copper hydroxide, Oxytetracycline, Acibenzolar-S-methyl"
    },
    "Tomato___Early_blight": {
        "Organic": "Copper fungicides, potassium bicarbonate, crop rotation",
        "Inorganic": "Chlorothalonil, Mancozeb, Azoxystrobin"
    },
    "Tomato___Late_blight": {
        "Organic": "Copper-based fungicides, proper spacing, resistant varieties",
        "Inorganic": "Chlorothalonil, Mancozeb, Dimethomorph"
    },
    "Tomato___Leaf_Mold": {
        "Organic": "Improve air circulation, neem oil, potassium bicarbonate",
        "Inorganic": "Chlorothalonil, Copper oxychloride, Mancozeb"
    },
    "Tomato___Septoria_leaf_spot": {
        "Organic": "Copper fungicides, proper spacing, remove infected leaves",
        "Inorganic": "Chlorothalonil, Mancozeb, Azoxystrobin"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "Organic": "Insecticidal soap, neem oil, predatory mites",
        "Inorganic": "Abamectin, Bifenthrin, Spiromesifen"
    },
    "Tomato___Target_Spot": {
        "Organic": "Copper fungicides, proper spacing, crop rotation",
        "Inorganic": "Chlorothalonil, Mancozeb, Pyraclostrobin"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "Organic": "Reflective mulch, beneficial insects, resistant varieties",
        "Inorganic": "Imidacloprid, Thiamethoxam (for vector control)"
    },
    "Tomato___Tomato_mosaic_virus": {
        "Organic": "Remove infected plants, sanitize tools, resistant varieties",
        "Inorganic": "No direct chemical control, focus on vector management"
    },
    "Tomato___healthy": {
        "Organic": "Balanced organic fertilization, adequate calcium, proper spacing",
        "Inorganic": "Balanced NPK fertilizers with calcium supplements"
    }
}

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Disease Recognition"])

# Prediction Page
if app_mode == "Disease Recognition":
    st.header("Plant Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image:
        st.image(test_image, use_column_width=True)
        
        if st.button("Predict"):
            result_index = model_prediction(test_image)
            predicted_disease = class_name[result_index]
            st.success(f"Disease: **{predicted_disease}**")

            # Show chemical treatments
            if predicted_disease in treatment_info:
                st.subheader("🧪 Suggested Treatment:")
                st.markdown(f"- **Organic**: {treatment_info[predicted_disease]['Organic']}")
                st.markdown(f"- **Inorganic**: {treatment_info[predicted_disease]['Inorganic']}")
            else:
                st.warning("No chemical treatment info available for this disease.")