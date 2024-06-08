import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import io

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

# Create an instance of the router
router = StreamlitRouter()

if 'file' not in st.session_state:
    st.session_state.file = None

if 'df' not in st.session_state:
    st.session_state.df = None

# Define your pages 
def home():
    select_file = st.file_uploader("")
    if select_file is not None:
       st.session_state.file = select_file.getvalue()

    button = st.button("Đọc file")
    if button:
        if st.session_state.file is not None:
            if select_file.name.endswith('.csv'):
                df = pd.read_csv(select_file)
                st.write(df)
                st.session_state.df = df  
                st.session_state.columns = df.columns.tolist()
                print(st.session_state.columns)
            elif select_file.name.endswith('.xlsx'):
                df = pd.read_excel(select_file)
                st.write(df)
                st.session_state.df = df  
                st.session_state.columns = df.columns.tolist()
                print(st.session_state.columns)
        else:
            st.warning('Vui lòng chọn dataset')

def xu_ly_du_lieu():
    if 'df' in st.session_state:
        btn1 = st.button("Kiểm tra Missing value")
        col1, col2 = st.columns(2)
        with col1:
            option_columns = st.selectbox('Chọn dữ liệu cần xử lý', st.session_state.df.columns)
        with col2:
            option_data = st.selectbox('Chọn cách xử lý', ['Xóa dữ liệu null', 'Lấy giá trị trung bình', 'Lấy theo tuần suất xuất hiện'])
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


    

def thong_tin_du_lieu():
    col1, col2, col3 = st.columns(3)
    with col1:
        btn1 = st.button("Thông tin")
    with col2:
        btn2 = st.button("10 hàng đầu")
    with col3:
        btn3 = st.button("10 hàng cuối")

    col4, col5 = st.columns(2)
    with col4:
        btn4 = st.button("Tổng quan")
    with col5:
        btn5 = st.button("Độ lệch chuẩn")

    if btn1:
        buffer = io.StringIO()
        st.session_state.df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
    if btn2:
        st.write(st.session_state.df.head(10))
    if btn3:
        st.write(st.session_state.df.tail(10))
    if btn4:
        st.write(st.session_state.df.describe())
    if btn5:
        for column in st.session_state.df.columns:
            if st.session_state.df[column].dtype in ['int64', 'float64']:
                std_dev = st.session_state.df[column].std()
                st.write(f"Độ lệch chuẩn của cột <span style='color:red'>{column}</span>: {std_dev}", unsafe_allow_html=True)

def bieu_do():
    st.title('📞 Contact')
    st.image("https://via.placeholder.com/800x300?text=Contact+Page", use_column_width=True)
    st.write('## Contact Us')
    st.write('If you have any questions or feedback, feel free to reach out to us.')

def mo_hinh():
    st.title('📞 Contact')
    st.image("https://via.placeholder.com/800x300?text=Contact+Page", use_column_width=True)
    st.write('## Contact Us')
    st.write('If you have any questions or feedback, feel free to reach out to us.')

# Add routes to the router
router.add_route('Home', home)
router.add_route('thong_tin_du_lieu', thong_tin_du_lieu)
router.add_route('xu_ly_du_lieu', xu_ly_du_lieu)
router.add_route('bieu_do', bieu_do)
router.add_route('mo_hinh', mo_hinh)

# Initialize session state for page selection
if 'page' not in st.session_state:
    st.session_state.page = 'Home'  # Default page

# Design the sidebar interface
st.sidebar.markdown("<h1 style='text-align: center; color: #ff6347;'>Streamlit App</h1>", unsafe_allow_html=True)

# Add full-width button style
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

# Define navigation buttons and update session state
if st.sidebar.button('🏠 Trang chủ'):
    st.session_state.page = 'Home'
if st.sidebar.button('📋 Xem dữ liệu'):
    st.session_state.page = 'thong_tin_du_lieu'
if st.sidebar.button('✔️ Sử lý dữ liệu'):
    st.session_state.page = 'xu_ly_du_lieu'
if st.sidebar.button('📈 Biểu đồ'):
    st.session_state.page = 'bieu_do'
if st.sidebar.button('📚 Mô hình'):
    st.session_state.page = 'mo_hinh'

# Display the page based on the session state
router.route(st.session_state.page)
