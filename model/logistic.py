from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
def logistic(d_f, bien_muc_tieu, bien_doc_lap):
    X = d_f[bien_doc_lap]
    y = d_f[bien_muc_tieu]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    accuracy = (y_pred == y_test).mean()
    
    return y_pred, y_test, accuracy