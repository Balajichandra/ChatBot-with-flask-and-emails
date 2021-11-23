import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from tensorflow.keras.models import load_model
model = load_model('chat_model.h5')
import json
import random
intents = json.loads(open('courses_intend.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

file1 = open("data.txt","a")
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    file1.write("Bot : ")
    file1.write(str(res))
    file1.write("\n")
    return res


from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    file1.write("User : ")
    file1.write(str(userText))
    file1.write("\n")
    return chatbot_response(userText)


from flask_mail import Mail, Message
app1 = Flask(__name__)
mail = Mail(app1) # instantiate the mail class
   
# configuration of mail
app1.config['MAIL_SERVER']='smtp.gmail.com'
app1.config['MAIL_PORT'] = 465
app1.config['MAIL_USERNAME'] = 'sendermailid@gmail.com'
app1.config['MAIL_PASSWORD'] = '*********'
app1.config['MAIL_USE_TLS'] = False
app1.config['MAIL_USE_SSL'] = True
mail = Mail(app1)
   
# message object mapped to a particular URL ‘/’
@app1.route("/")
def index():
   msg = Message(
                'Hello',
                sender ='sendermailid@gmail.com',
                recipients = ['receivermailid@gmail.com']
               )
   msg.body = 'Hello Flask message sent from Flask-Mail'
   with app.open_resource("data.txt") as fp:  
      msg.attach("data.txt", "data/txt",fp.read())  
   mail.send(msg)
   return 'Mail Sent'


  
if __name__ == "__main__":
    app.run()
    app1.run()
open('data.txt', 'w').close()    