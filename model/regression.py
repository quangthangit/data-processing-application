from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

def linearRegression(d_f, bien_muc_tieu, bien_doc_lap):

    X = d_f[bien_doc_lap]
    y = d_f[bien_muc_tieu]

    # Handle missing values with mean imputation
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    # Calculate R-squared
    accuracy = model.score(X_test, y_test)
    print(f"Độ chính xác là (R-squared): {accuracy * 100:.2f}%")
    
    return y_pred, y_test, accuracy
