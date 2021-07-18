import numpy as np
import pickle

saved_svc = open("./model_files/model_svc.pkl", "rb")
model_svc=pickle.load(saved_svc)

input_data = np.array([[5,2,6,4]])
prediction = model_svc.predict(input_data)
print(str(prediction))
