import streamlit as st
from sklearn.model_selection import train_test_split
st.set_option('deprecation.showPyplotGlobalUse', False)
import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

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

# Function MissingValue
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

# Function Outliers
def outliers():
    if st.session_state.df is not None and not st.session_state.df.empty:
        col1, col2 = st.columns(2)
        with col1:
            option_columns = st.selectbox('Chọn dữ liệu phân tích', st.session_state.df.columns)
        with col2:
            option_action = st.selectbox('Lựa chọn', ['Xóa outliers', 'Lấy trung bình outliers', 'Trực quan hóa outliers'])
        
        col3, col4 = st.columns(2)
        with col3:
            btn1 = st.button("Kiểm tra Outliers")
        with col4:
            btn2 = st.button("Xử lý dữ liệu")

        if btn1:
            outliers_indices = detect_outliers(st.session_state.df[option_columns])
            st.write("Kết quả phát hiện ngoại lệ:")
            st.write(outliers_indices)
        if btn2:
            if option_action == 'Xóa outliers':
                st.session_state.df = remove_outliers(st.session_state.df, option_columns)
                st.write("Xóa outliers")
            elif option_action == 'Lấy trung bình outliers':
                st.session_state.df = impute_outliers(st.session_state.df, option_columns)
                st.write("Lấy trung bình outliers")
            elif option_action == 'Trực quan hóa outliers':
                visualize_outliers(st.session_state.df, option_columns)

    else:
        st.warning("Vui lòng nhập dataset")

# Function Outliers
def visualize_outliers(df, column):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=column)
    plt.title(f'Box Plot of {column}')
    st.pyplot()

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

def one_hot_encoding() :
    if st.session_state.df is not None and not st.session_state.df.empty:
        option = st.selectbox("Chọn biến", st.session_state.columns, placeholder="Chọn biến")
        col1 ,col2 ,col3 = st.columns(3)
        with col1 :
            btn1 = st.button("Xem unique")
        with col2 :
            btn2 = st.button("Dùng One-Hot Encoding")
        with col3 :
            btn3 = st.button("Dùng Label Encoding")    

        if btn1 :
            st.write(st.session_state.df[option].unique())
        if btn2 :
            st.session_state.df[option] = label_encoder.fit_transform(st.session_state.df[option])
            st.write(st.session_state.df[option].unique())
        if btn3 :
            st.session_state.df[option] = label_encoder.fit_transform(st.session_state.df[option])
            st.write(st.session_state.df[option].unique())
# Function Duplicate
def duplicate() :
    st.title('📞 Duplicate')

router.add_route('missing_value', missing_value)
router.add_route('outliers', outliers)
router.add_route('duplicate', duplicate)
router.add_route('one_hot_encoding', one_hot_encoding)

if st.sidebar.button('🔍 MissingValue'):
    st.session_state.page = 'missing_value'
if st.sidebar.button('📈 Outliers'):
    st.session_state.page = 'outliers'
if st.sidebar.button('🔄 Duplicate'):
    st.session_state.page = 'duplicate'
if st.sidebar.button('One-Hot Encoding and Label Encoding'):
    st.session_state.page = 'one_hot_encoding'

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