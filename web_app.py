from flask import Flask,render_template,request
import pickle
import numpy as np
from model import TextNormalizer




app = Flask(__name__)
model = pickle.load(open("MultinomialNB_model2.pkl","rb"))

@app.route("/",methods=["GET"])
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    if request.method== "POST":
        text = str(request.form["user_input"])

        prediction = model.predict([[text]])
        if prediction == 1:
            output = "bad"
        elif prediction == 2:
            output = "okay"
        elif prediction == 3:
            output = "good"
        else:
            output = "great"

        return render_template("index.html", rating_prediction="The review rate is {}".format(output))

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host= "0.0.0.0", port=8080)