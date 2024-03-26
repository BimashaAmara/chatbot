# import random
# import nltk
# from nltk.stem import WordNetLemmatizer
# import json
# import pickle
# import numpy as np
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam
# from keras.regularizers import l2

# # Initialize NLTK's WordNet lemmatizer
# lemmatizer = WordNetLemmatizer()

# # Read intents from JSON file
# with open('intents.json', 'r') as file:
#     intents = json.load(file)

# # Initialize lists
# words = []
# classes = []
# documents = []
# ignore_words = ['?', '!']

# # Loop through intents and patterns to tokenize words, lemmatize, and preprocess data
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         # Tokenize each word
#         w = nltk.word_tokenize(pattern)
#         words.extend(w)
#         # Add documents in the corpus
#         documents.append((w, intent['tag']))
#         # Add unique tags to classes list
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# # Lemmatize, lowercase, and remove duplicates from words list
# words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# words = sorted(list(set(words)))
# classes = sorted(list(set(classes)))

# # Save words and classes to pickle files
# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(classes, open('classes.pkl', 'wb'))

# # Create training data
# training = []
# output_empty = [0] * len(classes)

# # Loop through documents to create bag of words and output row
# for doc in documents:
#     bag = []
#     pattern_words = doc[0]
#     pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
#     for w in words:
#         bag.append(1) if w in pattern_words else bag.append(0)
#     output_row = list(output_empty)
#     output_row[classes.index(doc[1])] = 1
#     training.append([bag, output_row])

# # Shuffle training data
# random.shuffle(training)

# # Split training data into features and labels
# X = np.array([item[0] for item in training])
# y = np.array([item[1] for item in training])

# # Split data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define model architecture
# model = Sequential()
# model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu', kernel_regularizer=l2(0.001)))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
# model.add(Dropout(0.5))
# model.add(Dense(len(classes), activation='softmax'))

# # Compile model
# model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# # Train model
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=5, verbose=1)

# # Save model
# model.save('chatbot_model.h5')

# print("Model completed and trained.")

import random
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# Initialize NLTK's WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Read intents from JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Initialize lists
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Loop through intents and patterns to tokenize words, lemmatize, and preprocess data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((w, intent['tag']))
        # Add unique tags to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lowercase, and remove duplicates from words list
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

# Loop through documents to create bag of words and output row
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle training data
random.shuffle(training)

# Split training data into features and labels
X = np.array([item[0] for item in training])
y = np.array([item[1] for item in training])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model architecture
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Define checkpoint to save only the best model
checkpoint = ModelCheckpoint('chatbot_model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=5, verbose=1, callbacks=[checkpoint])

model = load_model('chatbot_model.keras')
print("Model completed and trained.")
