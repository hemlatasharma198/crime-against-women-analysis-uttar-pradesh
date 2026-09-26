import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Crime Against Women",
    page_icon="🚨",
    layout="wide"
)


# =========================================================
# LOAD FILES
# =========================================================

df = pd.read_csv("up_women_crime.csv")

model = joblib.load("crime_prediction_model.pkl")

risk_profile = pd.read_csv("district_risk_profile.csv")

model_metrics = pd.read_csv("model_performance.csv")

prediction_comparison = pd.read_csv(
    "prediction_comparison.csv"
)

feature_importance = pd.read_csv(
    "feature_importance.csv"
)


# =========================================================
# CRIME CATEGORIES
# =========================================================

crime_cols = [
    "Rape",
    "Kidnapping_Abduction",
    "Dowry_Deaths",
    "Assault_to_outrage_her_modesty",
    "Insult_to_modesty_of_Women",
    "Cruelty_by_Husband_or_his_Relatives",
    "Importation_Girls"
]


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🚨 Crime Against Women")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📊 Crime Analysis",
        "🛡️ District Risk Profile",
        "🤖 Crime Prediction",
        "📈 Model Performance",
        "💡 Insights"
    ]
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.title("🚨 Crime Against Women")
    st.subheader("Uttar Pradesh Crime Analysis & Prediction")

    st.write(
        """
        This interactive application analyzes crime against women
        in Uttar Pradesh and uses Machine Learning to predict
        next-year total crime.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Records",
            len(df)
        )

    with col2:
        st.metric(
            "Districts",
            df["district"].nunique()
        )

    with col3:
        st.metric(
            "Years Covered",
            df["Year"].nunique()
        )

    st.divider()

    st.info(
        "Use the sidebar to explore crime analysis, "
        "district risk profiling, machine-learning predictions, "
        "model performance, and key insights."
    )


# =========================================================
# CRIME ANALYSIS
# =========================================================

elif page == "📊 Crime Analysis":

    st.title("📊 Crime Analysis")

    st.write(
        "Explore crime against women across districts and years "
        "in Uttar Pradesh."
    )

    st.divider()

    years = sorted(
        df["Year"].unique()
    )

    selected_year = st.selectbox(
        "Select Year",
        years
    )

    year_data = df[
        df["Year"] == selected_year
    ]

    st.subheader(
        f"Crime Statistics — {selected_year}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Crime",
            int(
                year_data["total_crime"].sum()
            )
        )

    with col2:

        st.metric(
            "Districts",
            year_data["district"].nunique()
        )

    with col3:

        st.metric(
            "Average Crime per District",
            round(
                year_data["total_crime"].mean(),
                2
            )
        )

    st.divider()

    crime_totals = (
        year_data[crime_cols]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    st.subheader(
        "Crime Category Distribution"
    )

    st.bar_chart(
        crime_totals
    )

    st.divider()

    st.subheader(
        "Top 10 Districts by Total Crime"
    )

    district_totals = (
        year_data
        .groupby("district")["total_crime"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    st.bar_chart(
        district_totals
    )


# =========================================================
# DISTRICT RISK PROFILE
# =========================================================

elif page == "🛡️ District Risk Profile":

    st.title("🛡️ District Crime Risk Profile")

    st.write(
        "Explore the historical crime burden, top crime categories, "
        "recent trends, and preventive focus areas for a selected district."
    )

    st.divider()

    district_list = sorted(
        risk_profile["District"].unique()
    )

    selected_district = st.selectbox(
        "Select District",
        district_list
    )

    district_data = risk_profile[
        risk_profile["District"] == selected_district
    ].iloc[0]

    # ---------------------------------------------------------
    # Main Indicators
    # ---------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Historical Crime Indicator",
            f"{district_data['Overall Risk Indicator']:.1f}/100"
        )

    with col2:

        st.metric(
            "Historical Burden",
            district_data["Historical Burden Band"]
        )

    with col3:

        trend = district_data[
            "Overall Trend Change %"
        ]

        if pd.isna(trend):

            trend_text = "Not Available"

        elif trend > 0:

            trend_text = f"↑ {trend:.1f}%"

        elif trend < 0:

            trend_text = f"↓ {abs(trend):.1f}%"

        else:

            trend_text = "Stable"

        st.metric(
            "Overall Trend",
            trend_text
        )

    st.divider()

    st.info(
        "The Historical Crime Indicator is a relative indicator "
        "based on reported crime counts in the latest available year. "
        "It is not a probability that a crime will occur."
    )

    st.divider()

    # ---------------------------------------------------------
    # Top 3 Crime Categories
    # ---------------------------------------------------------

    st.subheader(
        "🔎 Top 3 Crime Categories"
    )

    top_crimes = []

    for i in range(1, 4):

        crime = district_data[
            f"Top {i} Crime"
        ]

        indicator = district_data[
            f"Top {i} Indicator"
        ]

        trend_value = district_data[
            f"Top {i} Trend %"
        ]

        top_crimes.append(
            {
                "Crime Category": crime,
                "Relative Indicator": indicator,
                "Trend Change %": trend_value
            }
        )

    crime_table = pd.DataFrame(
        top_crimes
    )

    st.dataframe(
        crime_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ---------------------------------------------------------
    # Relative Indicators Chart
    # ---------------------------------------------------------

    st.subheader(
        "📊 Relative Crime Indicators"
    )

    indicator_chart = pd.DataFrame(
        {
            "Crime Category": [
                item["Crime Category"]
                for item in top_crimes
            ],
            "Indicator": [
                item["Relative Indicator"]
                for item in top_crimes
            ]
        }
    )

    indicator_chart = indicator_chart.set_index(
        "Crime Category"
    )

    st.bar_chart(
        indicator_chart
    )

    st.divider()

    # ---------------------------------------------------------
    # Preventive Focus Areas
    # ---------------------------------------------------------

    st.subheader(
        "🛡️ Preventive Focus Areas"
    )

    for i in range(1, 4):

        crime = district_data[
            f"Top {i} Crime"
        ]

        focus_text = district_data[
            f"Top {i} Preventive Focus"
        ]

        st.markdown(
            f"### {i}. {crime}"
        )

        focus_points = focus_text.split(
            " | "
        )

        for point in focus_points:

            st.write(
                f"• {point}"
            )

    st.divider()

    st.caption(
        "These focus areas are informational and are based on "
        "historical crime categories identified in the dataset."
    )


# =========================================================
# CRIME PREDICTION
# =========================================================

elif page == "🤖 Crime Prediction":

    st.title(
        "🤖 Next-Year Crime Prediction"
    )

    st.write(
        "Enter the previous year's district-level crime values "
        "to predict the next year's total crime."
    )

    st.divider()

    district_list = sorted(
        df["district"].unique()
    )

    selected_district = st.selectbox(
        "Select District",
        district_list
    )

    st.subheader(
        "Previous-Year Crime Data"
    )

    col1, col2 = st.columns(2)

    with col1:

        previous_rape = st.number_input(
            "Rape",
            min_value=0,
            value=0
        )

        previous_kidnapping = st.number_input(
            "Kidnapping / Abduction",
            min_value=0,
            value=0
        )

        previous_dowry = st.number_input(
            "Dowry Deaths",
            min_value=0,
            value=0
        )

        previous_assault = st.number_input(
            "Assault to Outrage Modesty",
            min_value=0,
            value=0
        )

    with col2:

        previous_insult = st.number_input(
            "Insult to Modesty",
            min_value=0,
            value=0
        )

        previous_cruelty = st.number_input(
            "Cruelty by Husband / Relatives",
            min_value=0,
            value=0
        )

        previous_importation = st.number_input(
            "Importation of Girls",
            min_value=0,
            value=0
        )

        previous_total = st.number_input(
            "Previous-Year Total Crime",
            min_value=0,
            value=0
        )

    st.divider()

    if st.button(
        "🔮 Predict Next-Year Crime"
    ):

        input_data = pd.DataFrame(
            [[
                previous_rape,
                previous_kidnapping,
                previous_dowry,
                previous_assault,
                previous_insult,
                previous_cruelty,
                previous_importation,
                previous_total
            ]],
            columns=[
                "previous_Rape",
                "previous_Kidnapping_Abduction",
                "previous_Dowry_Deaths",
                "previous_Assault_to_outrage_her_modesty",
                "previous_Insult_to_modesty_of_Women",
                "previous_Cruelty_by_Husband_or_his_Relatives",
                "previous_Importation_Girls",
                "previous_total_crime"
            ]
        )

        prediction = model.predict(
            input_data
        )[0]

        st.success(
            f"Predicted Next-Year Total Crime: **{prediction:.0f}**"
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📈 Model Performance":

    st.title(
        "📈 Model Performance"
    )

    st.write(
        "Performance evaluation of the Random Forest model "
        "used for next-year total crime prediction."
    )

    st.divider()

    st.subheader(
        "📌 Evaluation Metrics"
    )

    mae_value = model_metrics.loc[
        model_metrics["Metric"] == "MAE",
        "Value"
    ].iloc[0]

    rmse_value = model_metrics.loc[
        model_metrics["Metric"] == "RMSE",
        "Value"
    ].iloc[0]

    r2_value = model_metrics.loc[
        model_metrics["Metric"] == "R2 Score",
        "Value"
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "MAE",
            f"{mae_value:.2f}"
        )

    with col2:

        st.metric(
            "RMSE",
            f"{rmse_value:.2f}"
        )

    with col3:

        st.metric(
            "R² Score",
            f"{r2_value:.3f}"
        )

    st.info(
        "MAE and RMSE measure prediction error. "
        "R² indicates how much variation in the target "
        "is explained by the model."
    )

    st.divider()

    # ---------------------------------------------------------
    # Actual vs Predicted
    # ---------------------------------------------------------

    st.subheader(
        "📊 Actual vs Predicted Crime"
    )

    prediction_chart = prediction_comparison[
        [
            "Actual Crime",
            "Predicted Crime"
        ]
    ].copy()

    st.line_chart(
        prediction_chart
    )

    st.divider()

    # ---------------------------------------------------------
    # Feature Importance
    # ---------------------------------------------------------

    st.subheader(
        "🌳 Feature Importance"
    )

    importance_chart = feature_importance[
        [
            "Feature",
            "Importance"
        ]
    ].copy()

    importance_chart = importance_chart.sort_values(
        "Importance",
        ascending=True
    )

    importance_chart = importance_chart.set_index(
        "Feature"
    )

    st.bar_chart(
        importance_chart
    )

    st.write(
        "Feature importance shows which previous-year crime "
        "variables contributed most to the Random Forest predictions."
    )


# =========================================================
# INSIGHTS
# =========================================================

elif page == "💡 Insights":

    st.title(
        "💡 Key Insights"
    )

    st.write(
        "Important findings derived from the crime dataset "
        "and machine-learning analysis."
    )

    st.divider()

    # ---------------------------------------------------------
    # Dataset Overview
    # ---------------------------------------------------------

    st.subheader(
        "📊 Dataset Overview"
    )

    total_records = len(df)

    total_districts = df[
        "district"
    ].nunique()

    start_year = int(
        df["Year"].min()
    )

    end_year = int(
        df["Year"].max()
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Records",
            total_records
        )

    with col2:

        st.metric(
            "Districts",
            total_districts
        )

    with col3:

        st.metric(
            "Years Covered",
            f"{start_year}–{end_year}"
        )

    st.divider()

    # ---------------------------------------------------------
    # Crime Category Analysis
    # ---------------------------------------------------------

    st.subheader(
        "🔎 Crime Category Analysis"
    )

    crime_totals = (
        df[crime_cols]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    highest_crime = crime_totals.index[0]

    highest_crime_value = int(
        crime_totals.iloc[0]
    )

    st.write(
        f"**{highest_crime}** has the highest total "
        f"reported count among the crime categories in the "
        f"dataset, with **{highest_crime_value:,} cases**."
    )

    st.bar_chart(
        crime_totals
    )

    st.divider()

    # ---------------------------------------------------------
    # Year-wise Crime Trend
    # ---------------------------------------------------------

    st.subheader(
        "📅 Year-wise Crime Trend"
    )

    yearly_crime = (
        df.groupby("Year")[
            "total_crime"
        ]
        .sum()
        .sort_index()
    )

    highest_year = int(
        yearly_crime.idxmax()
    )

    highest_year_value = int(
        yearly_crime.max()
    )

    st.write(
        f"The highest aggregate reported crime count "
        f"in the dataset occurred in **{highest_year}**, "
        f"with **{highest_year_value:,} total cases**."
    )

    st.line_chart(
        yearly_crime
    )

    st.divider()

    # ---------------------------------------------------------
    # District-wise Analysis
    # ---------------------------------------------------------

    st.subheader(
        "📍 District-wise Analysis"
    )

    district_totals = (
        df.groupby("district")[
            "total_crime"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    highest_district = (
        district_totals.index[0]
    )

    highest_district_value = int(
        district_totals.iloc[0]
    )

    st.write(
        f"**{highest_district}** has the highest cumulative "
        f"reported crime count across the available years, "
        f"with **{highest_district_value:,} cases**."
    )

    st.bar_chart(
        district_totals.head(10)
    )

    st.divider()

    # ---------------------------------------------------------
    # Machine Learning Insight
    # ---------------------------------------------------------

    st.subheader(
        "🤖 Machine Learning Insight"
    )

    most_important_feature = (
        feature_importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .iloc[0]
    )

    st.write(
        f"The most influential feature in the Random Forest "
        f"model is **{most_important_feature['Feature']}**, "
        f"with an importance score of "
        f"**{most_important_feature['Importance']:.3f}**."
    )

    st.info(
        "The model uses previous-year district crime "
        "information to forecast next-year total crime."
    )

    st.divider()

    # ---------------------------------------------------------
    # District Risk Profile Insight
    # ---------------------------------------------------------

    st.subheader(
        "🛡️ District Risk Profiling"
    )

    st.write(
        "The application also provides a district-level "
        "historical crime burden indicator. It highlights "
        "comparatively higher crime categories, recent trends, "
        "and preventive focus areas."
    )

    st.caption(
        "The district indicator is a relative historical measure "
        "based on reported crime counts and is not a probability "
        "of future crime occurrence."
    )