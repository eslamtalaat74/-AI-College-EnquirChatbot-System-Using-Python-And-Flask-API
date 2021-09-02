
import nltk
import numpy
from numpy import *
from pandas import *
import pandas as pd 
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from flask import Flask , request
app = Flask(__name__)

import numpy


from data_preprocessing import data_preprocessing
training, output,labels, words, data = data_preprocessing(stemmer)


    
###### architecture file       

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(training,output,test_size=0.25)

# X_train.to_csv('training.csv')
# pd.DataFrame(X_train).to_csv('X_train.csv')
# pd.DataFrame(X_test).to_csv('X_test.csv')
# pd.DataFrame(y_train).to_csv('y_train.csv')
# pd.DataFrame(y_test).to_csv('y_test.csv')

# numpy.save("X_train.npy",X_train)
# numpy.save("X_test.npy",X_test)
# numpy.save("y_train.npy",y_train)
# numpy.save("y_test.npy",y_test)

# numpy.load("X_train.npy")

# print(X_train.shape,y_train.shape)
# print(X_test.shape,y_test.shape)
# print(X_train,y_train)
# print(X_test,y_test)



from init_model import init_model
model = init_model(X_train,y_train,X_test,y_test)



#####
######### data pre processing module
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)
print("hi")
print(len(X_train[0]))  
print(len(y_train[0]))  

#### main file   
from chat import  chat       
#chat(model,bag_of_words,labels,words,data)



@app.route('/')
def hello_world():
    return 'Hellsso, World!'

@app.route("/get")
def get_bot_reponse():
    userText = request.args.get('msg')
    return str(chat(model,bag_of_words,labels,words,data,userText))

if __name__ == "__main__":
    app.run(debug=True)