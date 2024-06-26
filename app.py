import random
import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

app = Flask(__name__, static_folder='templates')

ABSOLUTE_PATH = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(ABSOLUTE_PATH, 'words.pkl'), 'rb') as f:
    words = pickle.load(f)
with open(os.path.join(ABSOLUTE_PATH, 'classes.pkl'), 'rb') as f:
    classes = pickle.load(f)
model = load_model(os.path.join(ABSOLUTE_PATH, 'chatbot_model.keras'))

def get_chatbot_response(user_input):
    try:
        intent_index = predict_user_input(user_input)
        if intent_index is None:
            return "Sorry, I couldn't process your request."
        intent = classes[intent_index]
        return get_response(intent)  # return the string response
    except Exception as e:
        print(f"Error getting chatbot response: {e}")
        return "Sorry, an error occurred while processing your request."

def get_response(intent):
    with open(os.path.join(ABSOLUTE_PATH, 'intents.json'), 'r') as file:
        intents_json = json.load(file)
    for intent_data in intents_json['intents']:
        if intent_data['tag'] == intent:
            responses = intent_data['responses']
            if isinstance(responses, list):
                response = random.choice(responses)
            else:
                response = responses
            return response  # return a string response

def preprocess_input(user_input):
    user_input = user_input.lower()
    user_input = nltk.word_tokenize(user_input)
    user_input = ' '.join([lemmatizer.lemmatize(word) for word in user_input])
    return user_input

def predict_user_input(user_input):
    try:
        user_input = preprocess_input(user_input)
        user_input_bow = bow(user_input, words)
        user_input_bow = np.array(user_input_bow).reshape(1, -1)
        output = model.predict(user_input_bow)
        intent_index = np.argmax(output)
        return int(intent_index)  
    except Exception as e:
        print(f"Error predicting user_input: {e}")
        return None

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return bag

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.json['user_input']
        response = get_chatbot_response(user_input)
        return jsonify({'user_input': user_input, 'response': response})
    return render_template('chatbot.html')

@app.route('/get_response', methods=['GET'])
def get_response_route():
    return jsonify({'response': 'This is a static test response from /get_response endpoint.'})

@app.errorhandler(500)
def internal_error(error):
    return str(error), 500

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json['user_input']
    print(f"User input: {user_input}")
    response = get_chatbot_response(user_input)
    print(f"Response type: {type(response)}")
    print(f"Response value: {response}")
    return jsonify({'response': response})  # return the string response

if __name__ == '__main__':
    app.run(debug=True)
