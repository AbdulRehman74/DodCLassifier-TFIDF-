import streamlit as st
import joblib

# Load the trained model
model_data = joblib.load('text_classifier_model_with_TFIDF.joblib')

classifier = model_data['classifier']
tfidf_vectorizer = model_data['tfidf']
label_encoder = model_data['label_encoder']

def predict_category(document, tfidf_vectorizer, classifier, label_encoder):
    document = document.split()
    input_vector = tfidf_vectorizer.transform([' '.join(document)])
    predicted_label = classifier.predict(input_vector)
    predicted_category = label_encoder.inverse_transform(predicted_label)
    return predicted_category[0]

st.title("Text Classification App")

input_document = st.text_area("Enter a document (paste or type) and press Enter when done:")
if input_document.strip().lower() == "exit":
    st.stop()

if st.button("Predict"):
    predicted_category = predict_category(input_document, tfidf_vectorizer, classifier, label_encoder)
    st.write(f"Predicted Category: {predicted_category}")
