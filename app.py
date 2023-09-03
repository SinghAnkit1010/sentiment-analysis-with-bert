import numpy as np
import pandas as pd
import re
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from flask import Flask,render_template,request

preprocess_layer = hub.KerasLayer(r"bert_en_uncased_preprocess_3")
bert_layer = hub.KerasLayer(r"small_bert_bert_en_uncased_L-6_H-256_A-4_2",trainable = False)

text_input = tf.keras.layers.Input(shape = (),dtype = tf.string,name = "text")
text_preprocessed = preprocess_layer (text_input)
outputs = bert_layer(text_preprocessed)
net = tf.keras.layers.Dropout(0.2)(outputs["pooled_output"])
output = tf.keras.layers.Dense(units =1,activation = "sigmoid", name = "output")(net)
model = tf.keras.Model(inputs = [text_input],outputs = output)

model.load_weights("./Bert_model.h5")

class Sentiment:
  def data_clean(self,sentence):
    list = []
    for text in sentence:
      string = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z\t])|(\w+:\/\/\S+)"," ",text).lower().split()
      string = " ".join(string)
      list.append(string)
    return list
  def result(self,sentence):
    data = self.data_clean(sentence)
    Result = model.predict(data)
    score = []
    sentiment = []
    prediction = []
    for i in range(len(sentence)):
      dict = {}
      value = Result[i][0]
      if(value>0.6):
        sentiment.append("positive")
      elif(value>0.4):
        sentiment.append("neutral")
      else:
        sentiment.append("negative")
      score.append(value)
      dict["setence"] = sentence[i]
      dict["score"] = score[i]
      dict["sentiment"] = sentiment[i]
      prediction.append(dict)
    return prediction[0]
obj = Sentiment()
app = Flask(__name__,template_folder='template')
results = []

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/predict", methods = ["GET","POST"])
def predict():
   if request.method == 'POST':
    text = request.form['text_data']
    text = [text]
    predict = obj.result(text)
    results.append(predict)
    return render_template("index.html",analysis_results = results)

if __name__ == "__main__":
  app.run(debug=True)
