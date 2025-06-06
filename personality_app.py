import streamlit as st
import joblib
import pandas as pd

# Load models
model = joblib.load('personality_model.pkl')
stage_encoder = joblib.load('stage_fear_encoder.pkl')
drained_encoder = joblib.load('drained_encoder.pkl')

def predict_personality(time_alone, social_events, going_out, friends, posts, stage_fear, drained):
    stage_encoded = stage_encoder.transform([stage_fear])[0]
    drained_encoded = drained_encoder.transform([drained])[0]
    
    new_data = pd.DataFrame({
        'Time_spent_Alone': [time_alone],
        'Social_event_attendance': [social_events], 
        'Going_outside': [going_out],
        'Friends_circle_size': [friends],
        'Post_frequency': [posts],
        'Stage_fear_encoded': [stage_encoded],
        'Drained_encoded': [drained_encoded]
    })
    
    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0].max()
    
    return prediction, probability

st.title("🧠 Personality Predictor")

time_alone = st.slider("Hours spent alone daily", 0, 11, 4)
social_events = st.slider("Social events per month", 0, 10, 3)
going_out = st.slider("Going outside frequency", 0, 7, 3)
friends = st.slider("Friends circle size", 0, 15, 6)
posts = st.slider("Social media posts per week", 0, 10, 3)
stage_fear = st.selectbox("Stage fear?", ["Yes", "No"])
drained = st.selectbox("Drained after socializing?", ["Yes", "No"])

if st.button("Predict Personality"):
    result, confidence = predict_personality(time_alone, social_events, going_out, friends, posts, stage_fear, drained)
    st.success(f"**Prediction: {result}**")
    st.info(f"Confidence: {confidence:.1%}")
