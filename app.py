import streamlit as st
import numpy as np
import pickle

st.title("Obesity Predictor")

gender = st.selectbox(
    "Gender",
    ("Male", "Female"),
)

age = st.slider("How old are you?", 0, 100, 25)

weight = st.slider("How much do you weigh?", 0, 200, 25)

history = st.radio(
    "Does your family have a history of being overweight?",
    ["yes", "no"],
)

obesity_level = np.array([[gender, age, weight, history]])

with open('rf.pkl', 'rb') as f:
    model = pickle.load(f)

prediction = model.predict(obesity_level)
st.write(prediction)