import streamlit as st

import pandas as pd

import numpy as np

import joblib

import re

from pathlib import Path

import matplotlib.pyplot as plt

import seaborn as sns


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="Review Sentiment Analyzer",

    page_icon="💬",

    layout="wide"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(
    __file__
).resolve().parent


MODEL_PATH = (
    BASE_DIR /
    "models" /
    "best_model.pkl"
)


VECTORIZER_PATH = (
    BASE_DIR /
    "models" /
    "tfidf_vectorizer.pkl"
)


DATA_PATH = (
    BASE_DIR /
    "dataset" /
    "processed_reviews.csv"
)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    return model, vectorizer


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_PATH
    )

    return data


# =========================================================
# LOAD MODEL AND DATA
# =========================================================

model, vectorizer = load_model()

data = load_data()


# =========================================================
# HEADER
# =========================================================

st.title(
    "💬 Review Sentiment Analyzer"
)

st.write(
    """
    An NLP-based machine learning application
    that analyzes customer reviews and predicts
    whether the sentiment is Positive or Negative.
    """
)


st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "📌 Navigation"
)


page = st.sidebar.radio(

    "Choose a section",

    [
        "Single Prediction",
        "Batch Prediction",
        "Analytics",
        "About"
    ]
)


# =========================================================
# SINGLE PREDICTION
# =========================================================

if page == "Single Prediction":

    st.header(
        "🔍 Analyze a Review"
    )

    st.write(
        "Enter a customer review below."
    )


    review = st.text_area(

        "Customer Review",

        placeholder=(
            "Example: "
            "This product is amazing. "
            "I really love it!"
        ),

        height=180
    )


    analyze_button = st.button(

        "🚀 Analyze Sentiment",

        use_container_width=True
    )


    if analyze_button:

        if not review.strip():

            st.warning(
                "Please enter a review first."
            )

        else:

            # ---------------------------------------------
            # Clean review
            # ---------------------------------------------

            cleaned_review = clean_text(
                review
            )


            # ---------------------------------------------
            # TF-IDF transformation
            # ---------------------------------------------

            review_vector = (
                vectorizer.transform(
                    [cleaned_review]
                )
            )


            # ---------------------------------------------
            # Prediction
            # ---------------------------------------------

            prediction = model.predict(
                review_vector
            )[0]


            # ---------------------------------------------
            # Display result
            # ---------------------------------------------

            st.subheader(
                "Prediction Result"
            )


            if prediction == 1:

                st.success(
                    "😊 Positive Sentiment"
                )

            else:

                st.error(
                    "😞 Negative Sentiment"
                )


            # ---------------------------------------------
            # Probability
            # ---------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = (
                    model.predict_proba(
                        review_vector
                    )[0]
                )


                classes = model.classes_


                probability_df = pd.DataFrame({

                    "Sentiment": [

                        "Negative"
                        if value == 0
                        else "Positive"

                        for value in classes
                    ],

                    "Probability":
                        probabilities

                })


                st.subheader(
                    "📊 Prediction Probability"
                )


                st.bar_chart(

                    probability_df.set_index(
                        "Sentiment"
                    )
                )


                confidence = (
                    max(probabilities) * 100
                )


                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )


            elif hasattr(
                model,
                "decision_function"
            ):

                decision = (
                    model.decision_function(
                        review_vector
                    )
                )


                score = float(
                    np.asarray(
                        decision
                    ).ravel()[0]
                )


                st.metric(

                    "Decision Score",

                    round(
                        score,
                        4
                    )
                )


            # ---------------------------------------------
            # Processed review
            # ---------------------------------------------

            with st.expander(
                "View processed review"
            ):

                st.write(
                    cleaned_review
                )


# =========================================================
# BATCH PREDICTION
# =========================================================

elif page == "Batch Prediction":

    st.header(
        "📁 Batch Review Prediction"
    )


    st.write(
        """
        Upload a CSV file containing a column
        named `review`.
        """
    )


    uploaded_file = st.file_uploader(

        "Upload CSV file",

        type=["csv"]
    )


    if uploaded_file:

        batch_data = pd.read_csv(
            uploaded_file
        )


        st.subheader(
            "Uploaded Data"
        )


        st.dataframe(

            batch_data,

            use_container_width=True
        )


        if "review" not in batch_data.columns:

            st.error(
                "Your CSV must contain a "
                "'review' column."
            )


        else:

            if st.button(

                "🚀 Predict All Reviews",

                use_container_width=True
            ):

                cleaned_reviews = (

                    batch_data[
                        "review"
                    ]

                    .fillna("")

                    .apply(clean_text)
                )


                review_vectors = (
                    vectorizer.transform(
                        cleaned_reviews
                    )
                )


                predictions = model.predict(
                    review_vectors
                )


                batch_data[
                    "predicted_sentiment"
                ] = [

                    "Positive"
                    if value == 1
                    else "Negative"

                    for value in predictions
                ]


                st.success(
                    "Prediction completed!"
                )


                st.dataframe(

                    batch_data,

                    use_container_width=True
                )


                # -----------------------------------------
                # Distribution
                # -----------------------------------------

                st.subheader(
                    "📊 Sentiment Distribution"
                )


                sentiment_counts = (

                    batch_data[
                        "predicted_sentiment"
                    ]

                    .value_counts()
                )


                st.bar_chart(
                    sentiment_counts
                )


                # -----------------------------------------
                # Download
                # -----------------------------------------

                csv = batch_data.to_csv(
                    index=False
                )


                st.download_button(

                    label=(
                        "⬇️ Download Predictions"
                    ),

                    data=csv,

                    file_name=(
                        "sentiment_predictions.csv"
                    ),

                    mime="text/csv",

                    use_container_width=True
                )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "Analytics":

    st.header(
        "📊 Dataset Analytics"
    )


    # Convert labels to names

    sentiment_display = (
        data["sentiment"]
        .map({
            0: "Negative",
            1: "Positive"
        })
    )


    total_reviews = len(data)


    positive_reviews = (
        (data["sentiment"] == 1)
        .sum()
    )


    negative_reviews = (
        (data["sentiment"] == 0)
        .sum()
    )


    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------

    col1, col2, col3 = st.columns(3)


    col1.metric(

        "Total Reviews",

        f"{total_reviews:,}"
    )


    col2.metric(

        "😊 Positive",

        f"{positive_reviews:,}"
    )


    col3.metric(

        "😞 Negative",

        f"{negative_reviews:,}"
    )


    st.divider()


    # ---------------------------------------------
    # Sentiment chart
    # ---------------------------------------------

    st.subheader(
        "😊 Sentiment Distribution"
    )


    sentiment_counts = (
        sentiment_display.value_counts()
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    sns.barplot(

        x=sentiment_counts.index,

        y=sentiment_counts.values,

        ax=ax
    )


    ax.set_xlabel(
        "Sentiment"
    )


    ax.set_ylabel(
        "Number of Reviews"
    )


    ax.set_title(
        "Review Sentiment Distribution"
    )


    st.pyplot(fig)


    # ---------------------------------------------
    # Dataset
    # ---------------------------------------------

    st.subheader(
        "📋 Sample Reviews"
    )


    display_data = data.copy()


    display_data[
        "sentiment"
    ] = display_data[
        "sentiment"
    ].map({
        0: "Negative",
        1: "Positive"
    })


    st.dataframe(

        display_data.head(20),

        use_container_width=True
    )


    # ---------------------------------------------
    # Dataset information
    # ---------------------------------------------

    st.subheader(
        "ℹ️ Dataset Information"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            "**Rows:**",
            data.shape[0]
        )


        st.write(
            "**Columns:**",
            data.shape[1]
        )


    with col2:

        st.write(

            "**Missing Values:**",

            data.isnull().sum().sum()
        )


        st.write(

            "**Duplicate Rows:**",

            data.duplicated().sum()
        )


# =========================================================
# ABOUT
# =========================================================

elif page == "About":

    st.header(
        "ℹ️ About This Project"
    )


    st.write(
        """
        ### Review Sentiment Analyzer

        This application uses Natural Language
        Processing and Machine Learning to classify
        movie/customer reviews into Positive and
        Negative sentiment.
        """
    )


    st.subheader(
        "🧠 NLP Pipeline"
    )


    st.markdown(
        """
        **Raw Review**

        ↓

        **Text Cleaning**

        ↓

        **TF-IDF Vectorization**

        ↓

        **Machine Learning Model**

        ↓

        **Positive / Negative**
        """
    )


    st.subheader(
        "🤖 Models Compared"
    )


    st.markdown(
        """
        - Logistic Regression
        - Multinomial Naive Bayes
        - Linear SVM
        - Random Forest
        """
    )


    st.subheader(
        "📈 Evaluation Metrics"
    )


    st.markdown(
        """
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - Confusion Matrix
        """
    )


    st.subheader(
        "📦 Dataset"
    )


    st.write(
        f"""
        The application uses a dataset containing
        {len(data):,} labeled reviews.
        """
    )


    st.info(
        """
        This project demonstrates an end-to-end
        NLP machine learning workflow from data
        preprocessing to model deployment.
        """
    )