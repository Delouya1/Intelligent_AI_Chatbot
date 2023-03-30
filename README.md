# Chatbot using Neural Network

This project contains the code for a chatbot that uses a neural network to classify user input and generate responses. The chatbot is created using Python and utilizes the following libraries:

* numpy
* json
* tensorflow
* keras
* nltk
* pickle

## Files

### The project contains two files:

* training.py: 
contains the code for training the neural network model. 
This script reads in a JSON file (intents.json) containing a set of predefined patterns and outputs the corresponding tag for each pattern. 
The script tokenizes and lemmatizes the patterns, converts them to a bag of words representation, and trains a neural network model to classify the patterns.
* chatbot.py: 
contains the code for running the chatbot. This script loads the trained model and listens for user input. It tokenizes and lemmatizes the user input, converts it to a bag of words representation, and uses the trained model to predict the appropriate response. The script outputs the predicted response and waits for the next user input.
Usage

### To use the chatbot, simply run the chatbot.py script. The script will load the trained model and begin listening for user input.
Enter text input and press enter to get a response from the chatbot.

## Requirements

This project requires the following Python libraries to be installed:

* numpy
* json
* tensorflow
* keras
* nltk
* pickle

## Note

The intents.json file should contain a list of intents, where each intent contains a tag, a list of patterns, and a list of responses. The patterns and responses should be specified as strings. 
The intents.json file should be located in the same directory as the training.py and chatbot.py scripts.
