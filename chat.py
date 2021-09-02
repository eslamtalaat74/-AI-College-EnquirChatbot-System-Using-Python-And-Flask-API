import numpy 
import random
from time import sleep

def chat(model,bag_of_words,labels,words,data,userText):
    
   # print("Hi, How can i help you ?")
   # while True:
        inp = userText
      
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        if results[results_index] > 0.8:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            sleep(3)
            Bot = random.choice(responses)
            return(Bot)
        else:
           return (random.choice(("I don't understand!", "Sorry, can't understand you","What does that mean?", "explain please!","more clarification,please!","what do you mean by that?","I do not understand what you say..!")))
            
            # print("I don't understand!")