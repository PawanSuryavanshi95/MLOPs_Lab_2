from flask import Flask, request, jsonify

from joblib import dump, load



app = Flask(__name__)



model_path = "svm_gamma=0.001_C=5.joblib"

model = load(model_path)



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



@app.route("/predict", methods=['POST'])

def predict_digit():



    image = request.json['image']

    print("Image Loaded")

    

    predicted = model.predict([ image])



    return {"prediction":int(predicted[0])}



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