from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load data
x, y = load_iris(return_X_y=True)

# Train model
model = RandomForestClassifier()
model.fit(x, y)

# Save model
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# import pickle
# import pandas as pd
# import os

# df=pd.read_csv("data/iris.csv")
# x=df.drop(columns=['Species',"Id"])
# y=df['Species']
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
# model=RandomForestClassifier()
# model.fit(x_train,y_train)

# os.makedirs("model",exist_ok=True)
# with open("model/model.pkl","wb")as f:
#     pickle.dump(model,f)
# print("model trained and saved")