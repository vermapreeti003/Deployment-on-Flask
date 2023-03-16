from logging import debug
from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET','POST'])
def index():
    # if request.method == 'GET':
    #     data = 'Hello World!'
    return render_template('index.html')
    # return jsonify({'data': data})

@app.route('/predict/', methods=['GET'])
def predict():
    gender = request.args.get('gender')
    age = int(request.args.get('age'))
    head_size = int(request.args.get('head_size'))

    temp_gender = gender
    temp_age = age

    if gender.strip().lower() == 'male':
        gender = 1
    else:
        gender = 2

    if age <= 18:
        age = 2
    else:
        age = 1

    test_in = np.array([gender,age,head_size]).reshape(1,-1)
    pred_weight = model.predict(test_in)
    output = round(pred_weight[0], 2)
    return render_template('result.html', gender="Gender: {}".format(temp_gender), age="Age: {}".format(temp_age),
     head_size="Head Size: {}".format(head_size), prediction_text="Brain Weight is {} grams".format(output))
    # return jsonify({'Brain Weight': output})

if __name__ == "__main__":
    app.run(debug=True)