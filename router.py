import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import matplotlib.pyplot as plt

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

if 'file' not in st.session_state:
    st.session_state.file = None
if 'columns' not in st.session_state:
    st.session_state.columns = []
if 'df' not in st.session_state:
    st.session_state.df = None

# Function Home
def home():
    select_file = st.file_uploader("Chọn file")
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

# Function Overview
def overview():
    if st.session_state.df is not None and not st.session_state.df.empty:
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
    else:
        st.warning("Vui lòng nhập dataset")

router.add_route('Home', home)
router.add_route('overview', overview)

if 'page' not in st.session_state:
    st.session_state.page = 'Home'  

st.sidebar.markdown("<h1 style='text-align: center; color: #ff6347;'>Trang chủ</h1>", unsafe_allow_html=True)
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

if st.sidebar.button('📝 Nhập dataset'):
    st.session_state.page = 'Home'
if st.sidebar.button('📋 Xem dữ liệu'):
    st.session_state.page = 'overview'

router.route(st.session_state.page)
