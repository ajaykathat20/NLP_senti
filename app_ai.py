## NPL with 
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
import base64

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load Google Gemini Pro Vision API And get response

def get_gemini_repsonse(input,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([input,prompt])
    return response.text

##initialize our streamlit app Frontend 



st.set_page_config(page_title="Sentiment Analysis")

st.header("Sentiment Analysis App")
image_path="360_F_614349682_VcWlQuUJ7Wtk2ZguTRlvEdMJ7LxmNI2e.jpg"
st.image(image_path,use_column_width=True)
st.markdown("""
### Understand the emotions behind the words.😊

Text sentiment prediction is a powerful tool that can help you to understand the emotions and opinions
expressed in your text data.
This information can be used to improve your business in a number of ways




""")
input=st.text_input("Input Prompt: ",key="input")

submit=st.button("Analysis")

input_prompt="""
Task: Classify the sentiment of a given text input (sentence, paragraph, etc.) as either positive or negative.

Input: The model will receive a text string containing the user's input.

Output: The model will predict one of two labels:

Positive: The text expresses a positive sentiment.
Negative: The text expresses a negative sentiment.
Training Data:

The model will be trained on a dataset consisting of text samples labeled as positive or negative. This dataset should be large and diverse, covering a wide range of topics and writing styles.

Evaluation:

The model's performance will be evaluated on a separate test dataset. Metrics like accuracy can be used to assess how well the model classifies sentiment.

Additional Considerations:

Neutral Sentiment: While the prompt focuses on positive and negative, you can consider adding a third label (Neutral) for cases where the text expresses no clear sentiment.
Confidence Score: The model can output a confidence score alongside the predicted sentiment. This score indicates the model's certainty in its prediction.
"""

## If submit button is clicked

if submit:
    response=get_gemini_repsonse(input_prompt,input)
    st.subheader("The Response is")
    st.write(response)