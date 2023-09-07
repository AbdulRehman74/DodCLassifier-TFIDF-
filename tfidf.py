import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

with open('docs.json', 'r') as json_file:
    data = json.load(json_file)

categories = [item['category'] for item in data]
text_data = [item['document'] for item in data]

text_data = [text.split() for text in text_data]

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform([' '.join(text) for text in text_data])

label_encoder = LabelEncoder()
encoded_categories = label_encoder.fit_transform(categories)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, encoded_categories, test_size=0.3, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate precision, recall, F1 score, and support for each class
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:")
print(classification_rep)

def predict_category(document, tfidf_vectorizer, classifier, label_encoder):
    document = document.split()
    input_vector = tfidf_vectorizer.transform([' '.join(document)])
    predicted_label = classifier.predict(input_vector)
    predicted_category = label_encoder.inverse_transform(predicted_label)
    return predicted_category[0]

model_data = {
    'classifier': classifier,
    'tfidf': tfidf_vectorizer,
    'label_encoder': label_encoder,
}

joblib.dump(model_data, 'text_classifier_model_with_TFIDF.joblib')


while True:
    input_document = input("Enter a document (paste or type) and press Enter when done. Type 'exit' to finish: ")

    if input_document.strip().lower() == "exit":
        break

    predicted_category = predict_category(input_document, tfidf_vectorizer, classifier, label_encoder)
    print(f"Predicted Category: {predicted_category}")
