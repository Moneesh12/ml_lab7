import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv("DCT_mal.csv")

X = df.drop(columns=["LABEL"]) 
y = df["LABEL"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

















