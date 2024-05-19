from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import streamlit as st
import re
import base64

STOPWORDS = set(stopwords.words("english"))

 # Select the predictor to be loaded from Models folder
predictor = pickle.load(open(r"E:\NLP Poject\model_xgb.pkl", "rb"))
scaler = pickle.load(open(r"E:\NLP Poject\scaler.pkl", "rb"))
cv = pickle.load(open(r"E:\NLP Poject\countVectorizer.pkl", "rb"))

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background('E:/NLP Poject/download.jpg')

def prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    print(corpus)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]


    return "Positive" if y_predictions == 1 else "Negative"

st.header("Text Sentiment Predictor")
image_path="sentiment.png"
st.image(image_path,use_column_width=True)
st.markdown("""
### Understand the emotions behind the words.😊

Text sentiment prediction is a powerful tool that can help you to understand the emotions and opinions
expressed in your text data.
This information can be used to improve your business in a number of ways




""")

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://images.app.goo.gl/LFCobouKtT7oZ7Qv7")
    }
   .sidebar .sidebar-content {
        background: url("https://images.app.goo.gl/LFCobouKtT7oZ7Qv7")
    }
    </style>
    """,
    unsafe_allow_html=True
)

text = st.text_input("Review")

if(st.button("Predict")):
        with st.spinner('Please wait...'):
            st.write("Our Prediction")

            result=prediction(predictor,scaler,cv,text)
            st.success("Model is Predicting it's a resulr : {}".format(result))
            st.balloons()