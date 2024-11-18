import pandas as pd
import streamlit as st
import numpy as np
import pickle

df = pd.read_csv("obesity.csv")
with open("obesity.pkl", "wb") as f:
    pickle.dump(df, f)

st.title("Obesity Predictor")

gender = st.selectbox(
    "Gender",
    ("Male", "Female"),
)

age = st.slider("How old are you?", 0, 130, 25)

weight = st.slider("How much do you weigh?", 0, 130, 25)

history = st.radio(
    "Does your family have a history of being overweight?",
    ["yes", "no"],
)

obesity_level = np.array([[gender, age, weight, history]])

with open('obesity.pkl', 'wb') as f:
    model = pickle.load(f)

prediction = model.predict(obesity_level)
st.write(prediction)