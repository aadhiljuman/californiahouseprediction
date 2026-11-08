import streamlit as st
import pandas as pd
import pickle
import base64
import io
from PIL import Image, ImageFilter

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠")

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]


@st.cache_data
def get_blurred_image(image_file):
    img = Image.open(image_file).convert("RGB").filter(ImageFilter.GaussianBlur(6))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def set_background(image_file):
    try:
        encoded = get_blurred_image(image_file)
    except FileNotFoundError:
        return
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .block-container {{
        background: rgba(255, 255, 255, 0.85);
        padding: 2rem 3rem;
        border-radius: 15px;
        margin-top: 2rem;
    }}
    </style>
    """, unsafe_allow_html=True)


set_background("bg.jpg")


@st.cache_resource
def load_models():
    return pickle.load(open("models.pkl", "rb"))


@st.cache_data
def load_results():
    return pd.read_csv("results.csv")


models = load_models()
results = load_results()
best_name = results.iloc[0]["Model"]

page = st.sidebar.radio("Menu", ["Predict Price", "Model Comparison"])

# ---------------- PAGE 1: PREDICT ----------------
if page == "Predict Price":
    st.title("🏠 California House Price Predictor")
    st.write("Fill in the details of the area below and click **Predict** to get the estimated house price.")

    st.subheader("Neighbourhood details")
    col1, col2 = st.columns(2)
    with col1:
        med_inc = st.number_input("Median Income (in $10,000s)", 0.5, 15.0, 3.5,
                                  help="3.5 means $35,000 per year")
        house_age = st.number_input("House Age (years)", 1, 52, 28)
        population = st.number_input("Population of the block", 3, 10000, 1200)
        ave_occup = st.number_input("Average people per house", 0.5, 10.0, 3.0)
    with col2:
        ave_rooms = st.number_input("Average rooms per house", 1.0, 20.0, 5.4)
        ave_bedrms = st.number_input("Average bedrooms per house", 0.5, 5.0, 1.0)
        latitude = st.number_input("Latitude", 32.5, 42.0, 34.2,
                                   help="Between 32.5 and 42.0 (California)")
        longitude = st.number_input("Longitude", -124.4, -114.3, -118.5,
                                    help="Between -124.4 and -114.3 (California)")

    if st.button("Predict", type="primary"):
        data = pd.DataFrame([[med_inc, house_age, ave_rooms, ave_bedrms,
                              population, ave_occup, latitude, longitude]],
                            columns=FEATURES)

        pred = models[best_name].predict(data)[0]
        st.success(f"Predicted price: ${pred * 100000:,.0f}")
        st.caption(f"Predicted using {best_name}, the best performing model")

        with st.expander("See what the other models predict"):
            for name, pipe in models.items():
                if name != best_name:
                    st.write(f"{name}: ${pipe.predict(data)[0] * 100000:,.0f}")

# ---------------- PAGE 2: COMPARISON ----------------
else:
    st.title("📊 Model Comparison")

    st.subheader("Performance on test data")
    st.dataframe(results, hide_index=True)

    best = results.iloc[0]
    st.info(f"Best model: **{best['Model']}** (R² = {best['R2']:.4f})")

    st.subheader("R² score (higher is better)")
    st.bar_chart(results.set_index("Model")["R2"])

    st.subheader("Error (lower is better)")
    st.bar_chart(results.set_index("Model")[["MAE", "MSE"]])

    st.subheader("Feature importance (Random Forest)")
    rf = models["Random Forest"].named_steps["model"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    st.bar_chart(imp)
