import pandas as pd

def getcolumns_dataset(df) :
    print("Tên các cột trong tập dữ liệu là:", df.columns.tolist())
    return df.columns.tolist()