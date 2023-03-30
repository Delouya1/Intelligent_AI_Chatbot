import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:  # Loop through each sentence in the intents patterns
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  # Split the sentence into words and store them in a list
        words.extend(word_list)  # Taking the contents of the list and adding them to the words list
        documents.append((word_list, intent['tag']))  # Add the words and the tag to the documents list
        if intent['tag'] not in classes:  # Add the tag to the classes list
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]  # Remove duplicates and
# lemmatize the words
words = sorted(set(words))  # Sort the words alphabetically

classes = sorted(set(classes))  # Sort the classes alphabetically

pickle.dump(words, open('words.pkl', 'wb'))  # Save the words list to a pickle file
pickle.dump(classes, open('classes.pkl', 'wb'))  # Save the classes list to a pickle file

# ML part
# We need to represent the words in a numerical format  we can train the model on
training = []
output_empty = [0] * len(classes)  # Create an empty list with the length of the classes list

for document in documents:  # Loop through each sentence in the documents list
    bag = []  # Create an empty list
    word_patterns = document[0]  # Get the words from the sentence
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]  # Lemmatize the words and
    # convert them to lowercase
    for word in words:  # Loop through each word in the words list
        bag.append(1) if word in word_patterns else bag.append(0)  # If the word is in the sentence, add 1 to the
        # bag list, otherwise add 0

    output_row = list(output_empty)  # Create a copy of the output_empty list
    output_row[classes.index(document[1])] = 1  # Set the index of the tag to 1
    training.append([bag, output_row])  # Add the bag list and the output_row list to the training list

random.shuffle(training)  # Shuffle the training list
training = np.array(training)  # Convert the training list to a numpy array

train_x = list(training[:, 0])  # Get the first column of the training array
train_y = list(training[:, 1])  # Get the second column of the training array

# Create the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))  # Add the first layer
model.add(Dropout(0.5))  # Add dropout to prevent overfitting
model.add(Dense(64, activation='relu'))  # Add the second layer
model.add(Dropout(0.5))  # Add dropout to prevent overfitting
model.add(Dense(len(train_y[0]), activation='softmax'))  # Add the output layer

# Compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # Create the optimizer
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # Compile the model

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)  # Train the model
model.save('chatbot_model.h5', hist)  # Save the model
print('Done')


