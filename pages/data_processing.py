import streamlit as st
import pandas as pd
import model.logistic as logistic
import model.regression as regression
import model.knn_regression as knn_regression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
st.set_option('deprecation.showPyplotGlobalUse', False)

if 'file' not in st.session_state:
    st.session_state.file = None
if 'columns' not in st.session_state:
    st.session_state.columns = []
if 'df' not in st.session_state:
    st.session_state.df = None

class StreamlitRouter:
    def __init__(self):
        self.routes = {}
        
    def add_route(self, path, func):
        self.routes[path] = func
        
    def route(self, path):
        if path in self.routes:
            self.routes[path]()
        else:
            st.error('404 - Page Not Found')

router = StreamlitRouter()

def missing_value() :
    if  st.session_state.df is not None and not st.session_state.df.empty:
        col1, col2 = st.columns(2)
        with col1:
            option_columns = st.selectbox('Chọn dữ liệu cần xử lý', st.session_state.df.columns)
        with col2:
            option_data = st.selectbox('Chọn cách xử lý', ['Xóa dữ liệu null', 'Lấy giá trị trung bình', 'Lấy theo tuần suất xuất hiện'])
        
        col3, col4 = st.columns(2)
        with col3:
            btn1 = st.button("Kiểm tra Missing value")
        with col4:
            btn2 = st.button("Xử lý dữ liệu")

        if btn1:
            st.write("Số lượng giá trị thiếu trong mỗi cột:")
            st.write(st.session_state.df.isnull().sum())
        if btn2:
            if option_data == 'Xóa dữ liệu null':
                st.session_state.df = st.session_state.df.dropna(subset=[option_columns])
                st.write(st.session_state.df.isnull().sum())
            elif option_data == "Lấy giá trị trung bình" : 
                mean_values = st.session_state.df[option_columns].mean()
                st.session_state.df[option_columns].fillna(mean_values, inplace=True)
                st.write(st.session_state.df.isnull().sum())
            else :
               most_frequent_value = st.session_state.df[option_columns].mode()[0]
               st.session_state.df[option_columns].fillna(most_frequent_value, inplace=True)
               st.write(st.session_state.df.isnull().sum())
    else:
        st.warning("Vui lòng nhập dataset")
def outliers() :
    st.title('📞 outliers')
def duplicate() :
    st.title('📞 Duplicate')

router.add_route('missing_value', missing_value)
router.add_route('outliers', outliers)
router.add_route('duplicate', duplicate)

if st.sidebar.button('🏠 MissingValue'):
    st.session_state.page = 'missing_value'
if st.sidebar.button('📋 Outliers'):
    st.session_state.page = 'outliers'
if st.sidebar.button('📋 Duplicate'):
    st.session_state.page = 'duplicate'

if 'page' not in st.session_state:
    st.session_state.page = 'missing_value' 

st.markdown(
    """
    <style>
    div.stButton > button {
        width: 100%;
        margin: 5px 0;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
router.route(st.session_state.page)