import pickle
import numpy as np
from flask import Flask, request
from flasgger import Swagger

saved_svc = open("./model_files/model_svc.pkl", "rb")
model_svc = pickle.load(saved_svc)

saved_rfc = open("./model_files/model_rfc.pkl", 'rb')
model_rfc = pickle.load(saved_rfc)

ml_api = Flask(__name__)
swagger = Swagger(ml_api)
    
    
@ml_api.route('/predict_svc')
def predict_svc():
  """Endpoint to predict the species of Iris [0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'] using Support Vector Machine
  ---
  parameters:
    - name: sepal_length
      in: query
      type: number
      required: true
    - name: sepal_width
      in: query
      type: number
      required: true
    - name: petal_length
      in: query
      type: number
      required: true
    - name: petal_width
      in: query
      type: number
      required: true
  responses:
    200:
      description: The output values
  """
  sepal_length = request.args.get('sepal_length')
  sepal_width = request.args.get('sepal_width')
  petal_length = request.args.get('petal_length')
  petal_width = request.args.get('petal_width')

  input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
  prediction = model_svc.predict(input_data)
  print(str(prediction))
  return str(prediction[0])

@ml_api.route('/predict_rfc')
def predict_rfc():
  """Endpoint to predict the species of Iris [0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'] using Support Vector Machine
  ---
  parameters:
    - name: sepal_length
      in: query
      type: number
      required: true
    - name: sepal_width
      in: query
      type: number
      required: true
    - name: petal_length
      in: query
      type: number
      required: true
    - name: petal_width
      in: query
      type: number
      required: true
  responses:
    200:
      description: The output values
  """
  sepal_length = request.args.get('sepal_length')
  sepal_width = request.args.get('sepal_width')
  petal_length = request.args.get('petal_length')
  petal_width = request.args.get('petal_width')

  input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
  prediction = model_rfc.predict(input_data)
  print(str(prediction))
  return str(prediction[0])
  
if __name__ == '__main__':
      ml_api.run(debug=True)
