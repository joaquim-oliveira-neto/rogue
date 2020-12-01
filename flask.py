from flask import Flask, escape, request
from rogue.lib import predict

app = Flask(__name__)

@app.route('/')
def hello():
    # get param from http://127.0.0.1:8080/?name=value
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'

@app.route('/predict', methods=['POST'])
def predict_fare():
    inputs = request.get_json()
    # pipeline = get_model_from_gcp()
    # results = pipeline.predict(X)
    # return {"predictions": results}
    return 'OK'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
