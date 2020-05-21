# External Libraries
import torch
import streamlit as st

# Load the Model and PreTrained Weights
from model import architecture
from utils import helper

# Default Args
default_review = "The food was great. The service was quick and the staff was very polite."


@st.cache(allow_output_mutation=True)
def load_model_for_app():
    # Model Definition contains default params
    HAN = architecture.HanModel()
    # Pretrained Weights
    HAN.load_state_dict(torch.load('HAN.pth', map_location=torch.device('cpu')))
    HAN.eval()
    return HAN


pretrained_HAN = load_model_for_app()

st.title('Hierarchical Attention Networks(HAN) Demo')

st.header('Input')
input_text = st.text_input(label='Enter Review Here',
                           value=default_review)

if len(input_text) == 0:
    input_text = default_review

predicted_rating, highlighted_text = helper.predict(input_text, pretrained_HAN)

st.header('Output')
st.subheader('Prediction')
# Adding 1 because classes are from 0 to 4 for review stars 1 to 5
st.write("Predicted Rating is : {0}/5".format(predicted_rating+1))

st.subheader('Attention')
st.markdown(highlighted_text, unsafe_allow_html=True)
