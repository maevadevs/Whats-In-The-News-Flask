# This is the backend server for the webapp

# LIBRARIES
# *********
import os
import sklearn
import json
import boto3
import joblib
import torch
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from flask import Flask, request, redirect, url_for, flash, jsonify
from transformers import AutoTokenizer, AutoModelWithLMHead #AutoModelForCausalLM


# SECRETS
# *******
# load env variables
load_dotenv() # => True if no error

# Grab our constants and secrets
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_KEY_ACCESS = os.getenv("S3_SECRET_KEY_ACCESS")
S3_USER = os.getenv("S3_USER")
S3_REGION_NAME = os.getenv("S3_REGION_NAME")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
SERVER_LISTENING_IP=os.getenv("SERVER_LISTENING_IP")

# Additional custom stopwords for classification
CUSTOM_STOPWORDS = ["said", "say", "says"]


# HELPER FUNCTIONS
# ****************

from helpers import get_category_mapping, get_summary

# This process_text() function is required by the pickle_file model to be available here
# It has to be defined before the model is loaded
def process_text(text):
    """
    Preprocess a given text: 
        - Lowercase
        - Tokenize
        - Remove non-needed tokens
        - Lemmatize
        - Clean
    """

    # Convert to lowercase, replace newlines with spaces, strip whitespaces
    text = text.lower().strip()

    # Tokenize
    word_tokens = word_tokenize(text)
    # Convert to a numpy array
    word_tokens = np.array(word_tokens)

    # Keep only alphabetic characters
    is_alpha = list(map(str.isalpha, word_tokens))
    word_tokens = word_tokens[is_alpha]

    # Remove stopwords
    custom_stopwords = CUSTOM_STOPWORDS
    stop_words = set(stopwords.words("english") + custom_stopwords)
    is_not_stopword = list(map(lambda token: token not in stop_words, word_tokens))
    word_tokens = word_tokens[is_not_stopword]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    vectorize_lemmatizer = np.vectorize(lemmatizer.lemmatize)
    word_tokens = vectorize_lemmatizer(word_tokens)

    # Convert into a setence form
    sentence = " ".join(word_tokens)

    # Return final tokenized sentence
    return sentence


# MODELS
# ******
# Model For Classification
# Only download model if it does not exist in the current directory
try:
    # Read the downloaded file from disc
    print("Loading model from disc...")
    with open("./model.pkl", 'rb') as model_file:
        classification_model = joblib.load(model_file)
    print("Loading model from disc successful.\n")
except:
    # Grab stored model from S3
    print("Model currently not existing. Fallback to downloading from source...")
    s3 = boto3.resource(
        service_name='s3',
        region_name=S3_REGION_NAME,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_KEY_ACCESS
    )

    # Download the file to disc
    s3.Bucket(S3_BUCKET_NAME).download_file(Key='MultinomialLogisticRegression-TFIDF5000-Best-RandomizedSearch3CV.pkl', Filename='model.pkl')

    # Read the downloaded file from disc
    with open("./model.pkl", 'rb') as model_file:
        classification_model = joblib.load(model_file)
    print("Loading model from source successful.\n")


# Model for Summarization
# Initialize pre-trained tokenizer and model
summary_tokenizer = AutoTokenizer.from_pretrained("t5-base")
summary_model = AutoModelWithLMHead.from_pretrained("t5-base", return_dict=True)

# Download all the nltk-needed stuff
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Create a new webapp instance
# ****************************

app = Flask(__name__)


# Preset all needed files at server start
# ***************************************

# # For large files that needs to be downloaded from somewhere, download them here
# pickle_file = "./models/MultinomialLogisticRegression-TFIDF5000-Best-RandomizedSearch3CV.pkl"
# # pickle_file = boto3.s3().getfile(filepath)

# # Then, recreate the prediction model
# with open(pickle_file, 'rb') as file:
#     model = joblib.load(file)


# Define routes and their handlers
# ********************************

# Handler of the root
@app.route('/')
def root_handler():
    # There is nothing here: Move along
    return {
      "code": 404,
      "message": "Not found"
    }

# Handler for prediction and summarization
@app.route('/api/predict_summarize/', methods=["POST"])
def predict_summarize_handler():
    # Get the passed data
    data = request.get_json() # An array of strings

    # Predict our category
    predicted_num = classification_model.predict(data)[0]

    # Generate Summary
    summary = get_summary(data[0], summary_tokenizer, summary_model)

    # Get our category mapping to their number from the helpers
    category = get_category_mapping(predicted_num)

    # Return the result as JSON
    return json.dumps({
      "code": 200,
      "number": str(predicted_num),
      "category": category,
      "summary": summary,
    })

# If exectuable, run the app
if __name__ == '__main__':
    app.run(
      host=SERVER_LISTENING_IP, 
      debug=False # Remove this for production
    )

# To run the app:
# python3 app-hello.py
