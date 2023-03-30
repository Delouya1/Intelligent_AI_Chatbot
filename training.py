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

