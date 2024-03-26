import re
import json
import random

# Load intents from JSON file
def load_intents(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

INTENTS_FILE = 'intents.json'
intents = load_intents(INTENTS_FILE)

def chat(input_text, intents):
    input_text = input_text.lower()
    for intent_data in intents['intents']:
        # Ensure intent_data is a dictionary
        if isinstance(intent_data, dict):
            # Access patterns key if available
            if 'patterns' in intent_data:
                for pattern in intent_data['patterns']:
                    if re.search(r'\b' + pattern + r'\b', input_text, re.IGNORECASE):
                        return get_response(intent_data)
    # If no matching intent is found, return a default response
    return "I'm sorry, I didn't understand that. Could you please rephrase your question?"



def get_response(intent):
    if intent['tag'] in intent['responses']:
        responses = intent['responses'][intent['tag']]
        if isinstance(responses, str):
            return responses
        else:
            return random.choice(responses)
    else:
        return "I'm sorry, I didn't understand that. Could you please rephrase your question?"


# Uncomment the following lines if you want to run the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    else:
        print("Bot:", chat(user_input, intents))

