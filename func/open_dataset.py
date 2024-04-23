import pandas as pd
import os

def open_dataset(file_path) :
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.csv':
        print("File là file CSV.")
        df = pd.read_csv(file_path)
        return df
    elif file_extension == '.xlsx':
        df = pd.read_excel(file_path)
        return df
    else:
        print("File không phải là file CSV hoặc XLSX.")
