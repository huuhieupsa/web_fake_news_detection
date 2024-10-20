from flask import Flask, render_template, request
import asyncio
from model_predict import model_predict
from load_post import load_post

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url_post = request.form['url_post']
    text = asyncio.run(load_post(url_post))
    pred = model_predict(text)
    detection = "True"
    if pred == 0:
        detection = "Fake"
    return render_template('index.html', url_post=url_post, detection=detection)
    

# main driver function
if __name__ == '__main__':
    app.run()

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)