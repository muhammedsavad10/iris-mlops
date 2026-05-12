import pickle
import json
import numpy as np

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def model_fn(model_dir):
    with open(f"{model_dir}/model.pkl","rb")as f:
        model=pickle.load(f)
    return model

def input_fn(request_body,request_content_type):
    print("🔥 NEW VERSION OF inference.py IS RUNNING 🔥")
    if request_content_type=="application/json":
        if isinstance(request_body,(bytes,bytearray)):
            request_body=request_body.decode("utf-8")
        data=json.loads(request_body)
        return np.array(data)
    raise ValueError("unsupported content type")


def predict_fn(input_data,model):
    return model.predict(input_data)

def output_fn(prediction,content_type):
    labels = [CLASS_NAMES[p] for p in prediction]
    return json.dumps({"prediction":labels})