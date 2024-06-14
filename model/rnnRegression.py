from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
def rnn_regression(d_f, bien_muc_tieu, bien_doc_lap):

    X = d_f[bien_doc_lap]
    y = d_f[bien_muc_tieu]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = r2_score(y_test,y_pred)
    
    return  y_pred, y_test, accuracy
