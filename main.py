import streamlit as st
import pickle

# Loading the trained model and vectorizer
model = pickle.load(open("emotion.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Define a function to predict emotion and provide advice
def predict_emotion(text):
    # Vectorize the input text
    text_vector = vectorizer.transform([text])

    # Predict the emotion
    emotion_label = model.predict(text_vector)[0]

    # Map the label to the emotion 
    emotions = {
        0: "Sadness",
        1: "Joy",
        2: "Love",
        3: "Anger",
        4: "Fear"
    }

    #  advice or responses based on the predicted emotion
    responses = {
        0: "It's okay to feel sad. Take your time to process your emotions.",
        1: "That's wonderful! Keep spreading the joy!",
        2: "Love is a beautiful feeling. Cherish it!",
        3: "Take a deep breath. It might help to step back and reflect.",
        4: "It's normal to feel afraid sometimes. Remember, you're stronger than you think."
    }

    # Getting the  emotion and corresponding response
    emotion = emotions[emotion_label]
    response = responses[emotion_label]

    return emotion, response

# Streamlit App
st.title("EMOTION RECOGNITION FROM TEXT")

# Create a radio button to toggle between the about me and main pafe
page = st.radio("Select a Page:", ["About Me", "Emotion Detection"])

# About Me
if page == "About Me":
    st.header("About Me")
    st.write("**Name**: Sah Godh Sunil Kumar")  
    st.write("**Institution email**: S.k.sahgodh@wlv.ac.uk")  
    st.write("**Student Number**: 2124956")  
    st.write("**Course**: Bsc(Hons) Computer Science")  


# Emotion Detection page
elif page == "Emotion Detection":
    st.header("Emotion Detection from Text")

    user_input = st.text_input("Enter a statement to analyze:")

    #submit button
    if st.button("Submit"):
        if user_input:
            # Predict emotion
            emotion, advice = predict_emotion(user_input)

            # Display the results
            st.write(f"**Predicted Emotion**: {emotion}")
            st.write(f"**Advice/Message**: {advice}")
        else:
            st.write("Please enter a statement before submitting.")
