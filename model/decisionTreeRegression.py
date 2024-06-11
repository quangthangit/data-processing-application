from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
def decisionTreeRegression(d_f, bien_muc_tieu, bien_doc_lap):

    X = d_f[bien_doc_lap]
    y = d_f[bien_muc_tieu]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = (r2_score(y_test,y_pred))
    
    return model ,y_pred, y_test, accuracy 