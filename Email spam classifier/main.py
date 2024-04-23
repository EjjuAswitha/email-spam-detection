from flask import Flask, render_template, request
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model
pipe = pickle.load(open("Naive_model.pkl", "rb"))

# Initialize NLTK components for preprocessing
nltk.download("punkt")
lstem = LancasterStemmer()
tfidf_vec = TfidfVectorizer(stop_words='english')


@app.route('/', methods=["GET", "POST"])
def main_function():
    if request.method == "POST":
        # Get the input email text from the form
        text = request.form
        emails = text['email']

        # Preprocess the input email text
        email_processed = preprocess(emails)

        # Make prediction
        output = pipe.predict([email_processed])[0]

        return render_template("show.html", prediction=output)

    else:
        return render_template("index.html")


def preprocess(email):
    # Tokenize the email text
    tokens = word_tokenize(email)
    # Stem the tokens
    stemmed_tokens = [lstem.stem(token) for token in tokens]
    # Join the stemmed tokens back into text
    email_processed = ' '.join(stemmed_tokens)
    return email_processed


if __name__ == '__main__':
    app.run(debug=True)


