import random
import json
import pickle
import time
import webbrowser

import billboard

import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from sphinx.util import requests
from tensorflow import keras
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence):  # Bag of words
    sentence_words = clean_up_sentence(sentence)  # Tokenize the pattern
    bag = [0] * len(words)  # Bag of words - matrix of N words, vocabulary matrix
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    # filter out predictions below a threshold
    p = bow(sentence)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(return_list, intents_json):
    if len(return_list) == 0:
        tag = 'noanswer'
    else:
        tag = return_list[0]['intent']
    if tag == 'datetime':
        print(time.strftime("%A"))
        print(time.strftime("%d %B %Y"))
        print(time.strftime("%H:%M:%S"))

    if tag == 'google':
        query = input('Enter query...')
        webbrowser.open('https://www.google.com/search?q=' + query)

    if tag == 'weather':
        api_key = '987f44e8c16780be8c85e25a409ed07b'
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        city_name = input("Enter city name : ")
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url)
        x = response.json()
        print('Present temp.: ', round(x['main']['temp'] - 273, 2), 'celcius ')
        print('Feels Like:: ', round(x['main']['feels_like'] - 273, 2), 'celcius ')
        print(x['weather'][0]['main'])

    if tag == 'news':
        main_url = " https://newsapi.org/v2/top-headlines?country=in&apiKey=bc88c2e1ddd440d1be2cb0788d027ae2"
        open_news_page = requests.get(main_url).json()
        article = open_news_page["articles"]
        results = []

        for ar in article:
            results.append([ar["title"], ar["url"]])

        for i in range(10):
            print(i + 1, results[i][0])
            print(results[i][1], '\n')

    if tag == 'song':
        chart = billboard.ChartData('hot-100')
        print('The top 10 songs at the moment are:')
        for i in range(10):
            song = chart[i]
            print(song.title, '- ', song.artist)

    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if tag == i['tag']:
            result = random.choice(i['responses'])
    return result


print("Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
