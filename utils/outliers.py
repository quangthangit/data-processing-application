import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def detect_outliers(data, threshold=3):
    z_scores = (data - data.mean()) / data.std()
    outliers_indices = z_scores[abs(z_scores) > threshold].index
    return outliers_indices

def remove_outliers(df, column):
    outliers_indices = detect_outliers(df[column])
    df_cleaned = df.drop(outliers_indices)
    return df_cleaned

def impute_outliers(df, column):
    outliers_indices = detect_outliers(df[column])
    median_value = df[column].median()
    df.loc[outliers_indices, column] = median_value
    return df
