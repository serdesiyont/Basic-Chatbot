import pickle
import random
import json

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

import tensorflow
from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download('wordnet')
intents = json.loads(open('intents.json').read())


lemmatizer = WordNetLemmatizer()

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('chatbot.keras')


def clean_up(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def prediction(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR]

    results.sort(key=lambda x : x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ''

    for i in list_of_intents:
        if i['tags'] == tag:
            result = random.choice(i['responses'])
            break
    return result
print("Running")

while True:
    message = input("")
    ints = prediction(message)
    res = get_response(ints, intents)
    print(res)


