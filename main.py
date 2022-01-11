from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__,template_folder='template')

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("job.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    req = request.form
    telecom = req.get("telecom")
    logo = req.get("logo")
    faq = req.get("faq")
    type1 = req.get("type")
    exp = req.get("exp")
    edu = req.get("edu")
    job = req.get("job")
    int_features = [[int(telecom), int(logo), int(faq), int(type1), int(exp), int(edu), int(job)]]
    final = np.array(int_features)
    # print(int_features)
    # print(final)
    prediction = model.predict(final)
    output = prediction

    if output == int(1):
        return render_template('job.html', pred='The job posting is: FAKE')
    else:
        return render_template('job.html', pred='The job posting is: LEGITIMATE')


if __name__ == '__main__':
    app.run(debug=True)
