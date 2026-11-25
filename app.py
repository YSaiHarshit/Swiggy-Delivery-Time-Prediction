
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Swiggy Delivery Time Prediction",
    page_icon="🍔",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Main container */

.stMainBlockContainer {
    max-width: 850px !important;
    width: 95% !important;
    padding: 1.5rem 2rem !important;
    margin: auto !important;
}

/* Main title */

.main-title {
    font-size: 38px;
    font-weight: 700;
    line-height: 1.3;
    text-align: center;
    color: #ff6b35;
    padding: 10px 0;
    margin: 0;
}

/* Subtitle */

.subtitle {
    font-size: 21px;
    line-height: 1.4;
    padding: 5px 0;
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}

/* Section headings */

.section-title {
    font-size: 23px;
    font-weight: bold;
    margin-top: 15px;
    margin-bottom: 15px;
    color: #333333;
}

/* Input fields */

.stTextInput,
.stNumberInput,
.stSelectbox,
.stDateInput {
    width: 100%;
}

/* Result card */

.result-box {
    width: 100%;
    border: 2px solid #F0A36B;
    background: linear-gradient(
        135deg,
        #FFF0E5,
        #FFF0E5
    );
    color: #803B12;
    border-radius: 17px;
    padding: 25px 15px;
    text-align: center;
    box-sizing: border-box;
    margin: 20px auto;
}

/* Result heading */

.result-box h2 {
    color: #803B12;
    font-size: 27px;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Result value */

.result-box .result-value {
    font-size: 42px;
    font-weight: bold;
    color: #C45B18;
    margin: 5px 0;
}

/* Prediction button */

.stFormSubmitButton button {
    width: 100%;
    min-height: 52px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
    background-color: #ff6b35;
    color: white;
    border: none;
}

/* Dividers */

hr {
    margin-top: 20px;
    margin-bottom: 20px;
}

/* Responsive design */

@media(max-width: 768px) {

    .stMainBlockContainer {
        width: 100% !important;
        padding: 1rem !important;
    }

    .main-title {
        font-size: 27px;
    }

    .subtitle {
        font-size: 17px;
    }

    .section-title {
        font-size: 20px;
    }

    .result-box .result-value {
        font-size: 35px;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_package = joblib.load(
        "swiggy_delivery_without_pipeline.pkl"
    )

    return model_package


try:

    model_package = load_model()

    model = model_package["model"]

    num_imputer = model_package["num_imputer"]

    cat_imputer = model_package["cat_imputer"]

    encoder = model_package["encoder"]

    scaler = model_package["scaler"]

    numerical_columns = list(
        model_package["numerical_columns"]
    )

    categorical_columns = list(
        model_package["categorical_columns"]
    )

except Exception as e:

    st.error(f"❌ Model Loading Error: {e}")

    st.stop()


# ============================================================
# 4. APPLICATION TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🍔 Swiggy Delivery Time Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict Food Delivery Time Using Machine Learning</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 5. INPUT FORM
# ============================================================

with st.form("swiggy_form"):

    # ========================================================
    # RIDER DETAILS
    # ========================================================

    st.markdown(
        '<div class="section-title">👤 Rider Details</div>',
        unsafe_allow_html=True
    )

    rider_id = st.text_input(
        "Rider ID",
        value="INDORES13DEL02"
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30,
        step=1
    )

    ratings = st.number_input(
        "Ratings",
        min_value=1.0,
        max_value=5.0,
        value=4.5,
        step=0.1
    )

    st.divider()


    # ========================================================
    # LOCATION DETAILS
    # ========================================================

    st.markdown(
        '<div class="section-title">📍 Location Details</div>',
        unsafe_allow_html=True
    )

    restaurant_latitude = st.number_input(
        "Restaurant Latitude",
        value=17.385000,
        format="%.6f"
    )

    restaurant_longitude = st.number_input(
        "Restaurant Longitude",
        value=78.486700,
        format="%.6f"
    )

    delivery_latitude = st.number_input(
        "Delivery Latitude",
        value=17.400000,
        format="%.6f"
    )

    delivery_longitude = st.number_input(
        "Delivery Longitude",
        value=78.500000,
        format="%.6f"
    )

    city_name = st.text_input(
        "City Name",
        value="HYD"
    )

    city_type = st.selectbox(
        "City Type",
        [
            "Metropolitian",
            "Urban",
            "Semi-Urban"
        ]
    )

    distance = st.number_input(
        "Distance (km)",
        min_value=0.0,
        value=5.0,
        step=0.1
    )

    st.divider()


    # ========================================================
    # ORDER DETAILS
    # ========================================================

    st.markdown(
        '<div class="section-title">🍽️ Order Details</div>',
        unsafe_allow_html=True
    )

    weather = st.selectbox(
        "Weather",
        [
            "sunny",
            "stormy",
            "sandstorms",
            "cloudy",
            "windy",
            "fog"
        ]
    )

    traffic = st.selectbox(
        "Traffic",
        [
            "low",
            "medium",
            "high",
            "jam"
        ]
    )

    type_of_order = st.selectbox(
        "Type of Order",
        [
            "Buffet",
            "Drinks",
            "Meal",
            "Snack"
        ]
    )

    type_of_vehicle = st.selectbox(
        "Type of Vehicle",
        [
            "bicycle",
            "electric_scooter",
            "motorcycle",
            "scooter"
        ]
    )

    vehicle_condition = st.number_input(
        "Vehicle Condition",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )

    multiple_deliveries = st.number_input(
        "Multiple Deliveries",
        min_value=0,
        max_value=5,
        value=0,
        step=1
    )

    st.divider()


    # ========================================================
    # OTHER DETAILS
    # ========================================================

    st.markdown(
        '<div class="section-title">🚦 Other Details</div>',
        unsafe_allow_html=True
    )

    festival = st.selectbox(
        "Festival",
        [
            "No",
            "Yes"
        ]
    )

    pickup_time_minutes = st.number_input(
        "Pickup Time (Minutes)",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

    order_time_hour = st.number_input(
        "Order Time Hour",
        min_value=0,
        max_value=23,
        value=12,
        step=1
    )

    order_date = st.date_input(
        "Select Order Date"
    )

    order_time_of_day = st.selectbox(
        "Order Time of Day",
        [
            "morning",
            "afternoon",
            "evening",
            "night"
        ]
    )

    st.divider()


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    submitted = st.form_submit_button(
        "🚀 Predict Delivery Time",
        use_container_width=True
    )


# ============================================================
# 6. PREDICTION
# ============================================================

if submitted:

    try:

        # ====================================================
        # DATE FEATURES
        # ====================================================

        date = pd.Timestamp(order_date)

        order_day = date.day

        order_month = date.month

        order_day_of_week = date.strftime(
            "%A"
        ).lower()

        is_weekend = (
            1 if date.dayofweek >= 5 else 0
        )


        # ====================================================
        # CREATE INPUT DATA
        # ====================================================

        input_data = pd.DataFrame({

            "rider_id": [rider_id],

            "age": [age],

            "ratings": [ratings],

            "restaurant_latitude": [
                restaurant_latitude
            ],

            "restaurant_longitude": [
                restaurant_longitude
            ],

            "delivery_latitude": [
                delivery_latitude
            ],

            "delivery_longitude": [
                delivery_longitude
            ],

            "weather": [weather],

            "traffic": [traffic],

            "vehicle_condition": [
                vehicle_condition
            ],

            "type_of_order": [type_of_order],

            "type_of_vehicle": [type_of_vehicle],

            "multiple_deliveries": [
                multiple_deliveries
            ],

            "festival": [festival],

            "city_type": [city_type],

            "city_name": [city_name],

            "order_day": [order_day],

            "order_month": [order_month],

            "order_day_of_week": [
                order_day_of_week
            ],

            "is_weekend": [is_weekend],

            "pickup_time_minutes": [
                pickup_time_minutes
            ],

            "order_time_hour": [
                order_time_hour
            ],

            "order_time_of_day": [
                order_time_of_day
            ],

            "distance": [distance]

        })


        # ====================================================
        # COLUMN ORDER
        # ====================================================

        required_columns = (
            numerical_columns + categorical_columns
        )

        missing_columns = [
            col for col in required_columns
            if col not in input_data.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Missing model input columns: {missing_columns}"
            )

        input_data = input_data[required_columns]


        # ====================================================
        # NUMERICAL IMPUTATION
        # ====================================================

        input_num = num_imputer.transform(
            input_data[numerical_columns]
        )


        # ====================================================
        # CATEGORICAL IMPUTATION
        # ====================================================

        input_cat = cat_imputer.transform(
            input_data[categorical_columns]
        )


        # ====================================================
        # ENCODING
        # ====================================================

        input_cat_encoded = encoder.transform(
            input_cat
        )

        if hasattr(input_cat_encoded, "toarray"):

            input_cat_encoded = (
                input_cat_encoded.toarray()
            )


        # ====================================================
        # SCALING
        # ====================================================

        input_num_scaled = scaler.transform(
            input_num
        )


        # ====================================================
        # FINAL DATA
        # ====================================================

        input_final = np.hstack(
            [
                input_num_scaled,
                input_cat_encoded
            ]
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        prediction = model.predict(
            input_final
        )

        predicted_time = float(
            prediction[0]
        )


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.success(
            "🎉 Prediction generated successfully!"
        )


        # ====================================================
        # FULL-WIDTH RESULT CARD
        # ====================================================

        st.markdown(
            f"""
       <div class ="result-box">
        <h2>🛵 Estimated Delivery Time</h2>
        <p class="result-value">{predicted_time:.2f} Minutes</p>
       </div>
        """,

        unsafe_allow_html=True
        )

        # ====================================================
        # INPUT DATA SUMMARY
        # ====================================================

        with st.expander("📋 View Input Data"):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            f"❌ Prediction Error: {e}"
        )


# ============================================================
# 7. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:gray;
        padding:15px;
        font-size:14px;
    ">
        🍔 Swiggy Delivery Time Prediction |
        Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
