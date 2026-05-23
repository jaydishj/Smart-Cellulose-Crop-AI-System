import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Smart Cellulose AI",
    page_icon="🌿",
    layout="wide"
)

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.main { background-color:#f4f9f4; }
.title { font-size:40px; font-weight:bold; color:#1b5e20; }
.subtitle { font-size:18px; color:#555; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌱 Smart Cellulose Crop AI System  </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find suitable cellulose crops | பொருத்தமான மரத்தாது பயிர்கள்</div>', unsafe_allow_html=True)
st.divider()

# =========================================================
# 100 CROPS DATABASE
# =========================================================
crops = [
"Eucalyptus","Teak","Pine","Spruce","Fir","Cedar","Oak","Birch","Poplar","Willow",
"Acacia","Maple","Beech","Ash","Alder","Mahogany","Rubber Tree","Neem","Sal","Casuarina",
"Cotton","Jute","Hemp","Flax","Sisal","Kenaf","Ramie","Abaca","Coir","Banana Fiber",
"Pineapple Fiber","Agave","Sunn Hemp","Roselle","Mesta","Urena","Date Palm Fiber","Sabai Grass",
"Rice Straw","Wheat Straw","Barley Straw","Oat Straw","Maize Stalk","Sorghum Stalk",
"Sugarcane Bagasse","Corn Husk","Rice Husk","Corn Cob","Paddy Straw",
"Bamboo","Giant Bamboo","Moso Bamboo","Napier Grass","Miscanthus","Switchgrass",
"Reed Grass","Vetiver","Giant Reed","Sudan Grass","Bermuda Grass","Guinea Grass",
"Brachiaria","Setaria","Energy Cane","Wild Sorghum",
"Coconut Husk","Banana Pseudostem","Oil Palm Frond","Oil Palm Empty Bunch",
"Coffee Husk","Tea Waste","Citrus Peel","Mango Peel","Jackfruit Waste",
"Water Hyacinth","Duckweed","Algae","Seaweed","Typha"
]

# =========================================================
# CATEGORY
# =========================================================
def get_category(crop):
    wood = ["Eucalyptus","Teak","Pine","Spruce","Fir","Cedar","Oak","Birch","Poplar","Willow","Acacia","Maple","Beech","Ash","Alder","Mahogany","Neem","Sal","Casuarina"]
    fiber = ["Cotton","Jute","Hemp","Flax","Sisal","Kenaf","Ramie","Abaca","Coir","Banana Fiber","Pineapple Fiber"]
    residue = ["Rice Straw","Wheat Straw","Barley Straw","Oat Straw","Maize Stalk","Sorghum Stalk","Sugarcane Bagasse","Corn Husk","Rice Husk","Corn Cob","Paddy Straw"]
    grass = ["Bamboo","Napier Grass","Miscanthus","Switchgrass","Giant Reed","Energy Cane","Wild Sorghum"]
    aquatic = ["Water Hyacinth","Duckweed","Algae","Seaweed","Typha"]

    if crop in wood: return "Wood Biomass | மர அடிப்படையியல்"
    if crop in fiber: return "Fiber Crop | நார் பயிர்"
    if crop in residue: return "Agricultural Residue | விவசாய கழிவு"
    if crop in grass: return "Fast Growing Grass | வேகமாக வளரும் புல்"
    if crop in aquatic: return "Aquatic Biomass | நீர்வாழ் உயிர் பொருள்"
    return "General Biomass | பொது உயிர் பொருள்"

# =========================================================
# DETAILS
# =========================================================
def get_details(category):
    if "Wood" in category:
        return ("₹80k–₹2L/acre", "5–10 years | 5–10 வருடம்", "Very High | மிக உயர்ந்த", "Paper, timber, nanocellulose", "Low | குறைவு")
    elif "Fiber" in category:
        return ("₹40k–₹1.2L/acre", "3–6 months | 3–6 மாதம்", "High | உயர்ந்த", "Textile, rope, bioplastic", "Low | குறைவு")
    elif "Residue" in category:
        return ("₹20k–₹70k/acre", "3–6 months | 3–6 மாதம்", "Medium | நடுத்தர", "Biofuel, paper pulp", "Medium | நடுத்தர")
    elif "Grass" in category:
        return ("₹60k–₹1.5L/acre", "3–12 months | 3–12 மாதம்", "Very High | மிக உயர்ந்த", "Bioenergy, cellulose fiber", "Low | குறைவு")
    elif "Aquatic" in category:
        return ("₹30k–₹90k/acre", "1–3 months | 1–3 மாதம்", "Medium | நடுத்தர", "Biofertilizer, cellulose extraction", "Medium | நடுத்தர")
    else:
        return ("₹25k–₹80k/acre", "3–8 months | 3–8 மாதம்", "Medium | நடுத்தர", "General use", "Medium | நடுத்தர")

# =========================================================
# DATASET
# =========================================================
@st.cache_data
def load_data():
    return pd.DataFrame({
        "Soil_Type": ["Loamy","Clay","Sandy","Loamy","Clay","Sandy","Loamy","Clay","Sandy","Loamy"],
        "Rainfall": [1200,900,700,1500,1100,600,1400,1000,750,1300],
        "Temp": [28,32,35,27,30,36,26,31,34,29],
        "Humidity": [80,65,55,85,70,50,88,68,58,82],
        "Region": ["Western Ghats","Dry","Semi-Arid","Forest","River","Dry","Hill","Wet","Semi-Arid","Forest"],
        "Crop": ["Bamboo","Kenaf","Hemp","Bamboo","Jute","Hemp","Bamboo","Jute","Sugarcane","Banana Fiber"]
    })

df = load_data()

soil_enc = LabelEncoder()
region_enc = LabelEncoder()
crop_enc = LabelEncoder()

df["Soil_Type"] = soil_enc.fit_transform(df["Soil_Type"])
df["Region"] = region_enc.fit_transform(df["Region"])
df["Crop"] = crop_enc.fit_transform(df["Crop"])

X = df.drop("Crop", axis=1)
y = df["Crop"]

@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# =========================================================
# SIDEBAR (BILINGUAL INPUT)
# =========================================================
st.sidebar.header("🌍 நிலைமைகள் | Farm Conditions")

soil = st.sidebar.selectbox("மண் | Soil Type", ["Loamy","Clay","Sandy"])
rain = st.sidebar.slider("மழை | Rainfall (mm)", 500, 2500, 1200)
temp = st.sidebar.slider("வெப்பநிலை | Temperature (°C)", 10, 45, 28)
hum = st.sidebar.slider("ஈரப்பதம் | Humidity (%)", 20, 100, 70)
region = st.sidebar.selectbox("பகுதி | Region", ["Western Ghats","Dry","Semi-Arid","Forest","River","Hill","Wet"])

# =========================================================
# INPUT ARRAY
# =========================================================
input_data = np.array([[
    soil_enc.transform([soil])[0],
    rain,
    temp,
    hum,
    region_enc.transform([region])[0]
]])

# =========================================================
# PREDICTION
# =========================================================
if st.button("🌿 பரிந்துரை செய்யவும் | Recommend Crop", use_container_width=True):

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data).max()

    crop = crop_enc.inverse_transform([pred])[0]
    category = get_category(crop)
    profit, duration, yield_lvl, use, risk = get_details(category)

    # Tamil crop names
    tamil_map = {
        "Bamboo": "மூங்கில்",
        "Kenaf": "கெனாஃப்",
        "Hemp": "சணல்",
        "Jute": "சணற்பை",
        "Sugarcane": "கரும்பு",
        "Banana Fiber": "வாழை நார்"
    }

    tamil_crop = tamil_map.get(crop, crop)

    st.success(f"🌱 பயிர் | Crop: {crop} ({tamil_crop})")

    st.info(f"📊 நம்பிக்கை | Confidence: {round(prob*100,2)}%")
    st.info(f"📦 வகை | Category: {category}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("💰 லாபம் | Profit", profit)

    with col2:
        st.metric("⏳ காலம் | Duration", duration)

    with col3:
        st.metric("🌾 உற்பத்தி | Yield", yield_lvl)

    st.write("🧪 பயன்பாடு | Use:", use)
    st.write("⚠ ஆபத்து | Risk:", risk)

    if prob > 0.85:
        st.success("🔥 சிறந்த தேர்வு | Excellent crop selection")
    elif prob > 0.7:
        st.info("👍 நல்ல தேர்வு | Good choice")
    else:
        st.warning("⚠ மிதமான பொருத்தம் | Moderate suitability")

# =========================================================
# FEATURE IMPORTANCE
# =========================================================
st.divider()
st.subheader("📊 அம்ச முக்கியத்துவம் | Feature Importance")

st.bar_chart(pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).set_index("Feature"))

st.divider()
st.markdown("🌿 Smart Cellulose AI System | நிலைத்த வேளாண்மை AI")
