import streamlit as st
import pickle
import numpy as np

# Set page config
st.set_page_config(page_title="Iris Flower Predictor 🌸", page_icon="🌸", layout="centered")

# Apply custom CSS for background and fonts
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    h1 {
        text-align: center;
        color: #4B0082;
    }
    .stButton>button {
        color: white;
        background-color: #4B0082;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
model = pickle.load(open('best_model.pkl', 'rb'))

# Title
st.title('🌸 Iris Flower Species Prediction')
st.write("Fill the flower measurements below:")

# Input sliders
with st.container():
    sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.8)
    sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
    petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.35)
    petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.3)

# Predict button
if st.button('Predict Species 🚀'):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    
    species = ['Setosa 🌱', 'Versicolor 🌼', 'Virginica 🌺']
    st.success(f'🎯 The predicted species is: **{species[prediction[0]]}**')
