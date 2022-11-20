from flask import Flask, request, jsonify

from joblib import dump, load

import os

app = Flask(__name__)

model_path = "models/svm_gamma=0.001_C=0.2.joblib"

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/sum", methods=['POST'])
def sum():

    print(request.json)
    x = request.json['X']
    y = request.json['Y']

    z = x+y


    return jsonify({"sum":z})


# endsem Q5
@app.route("/predict", methods=['POST'])
def predict_digit():

    images = request.json['images']
    model_name = request.json['model_name']	

    print("Images Loaded")

    mx = -1

    if model_name == "":
        for file in os.listdir("results"):
            f = open("results/" + file,"r+")
            content = f.read()
            f1_score = content.split('\n')[1].split(' ')[2]
            if mx < float(f1_score):
                mx = f1_score
                model_path = content.split('\n')[2].split(' ')[3]
    else:
        model_path = 'models/' + model_name

    model = load(model_path)

    predicted = model.predict(images)

    return {"prediction":str(predicted)}


# Quiz 4 Route : Predicts and checks whether two images have same label or not
@app.route("/check_label", methods=['POST'])
def check_label():

    image1 = request.json['image1']

    image2 = request.json['image2']

    print("Images Loaded")

    predicted = model.predict([ image1, image2 ])

    out = "Labels are same"

    if predicted[0] != predicted[1]:

        out = "Labels are different"

    return {"Result": out, "label1": int(predicted[0]), "label1": int(predicted[1])}


if __name__ == '__main__':

  print(__name__) 

  app.run(debug=True, host='0.0.0.0',port=5000)