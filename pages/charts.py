import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Function Univariate
def univariate():
    if st.session_state.df is not None and not st.session_state.df.empty:
        option_columns = st.selectbox('Chọn biến cần thống kê', st.session_state.df.columns)
    
        st.subheader(f"Biểu đồ cho cột {option_columns}")
        fig, ax = plt.subplots()

        column_data = st.session_state.df[option_columns]
        unique_values = column_data.nunique()

        if unique_values <= 50:
            unique_counts = column_data.value_counts()
            sns.barplot(x=unique_counts.index, y=unique_counts.values, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_xlabel('Giá trị dữ liệu')
            ax.set_ylabel('Số lượng')
        else:
            binned_data = pd.cut(column_data, bins=10)  
            counts = binned_data.value_counts().sort_index()
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_xlabel('Khoảng giá trị')
            ax.set_ylabel('Số lượng')
        
        st.pyplot(fig)
    else:
        st.warning("Vui lòng nhập dataset")

# Function Multivariate
def multivariate():
    if st.session_state.df is not None and not st.session_state.df.empty:
        col1, col2 = st.columns(2)
        with col1:
            option_columns1 = st.selectbox('Chọn biến cần thống kê thứ 1', st.session_state.df.columns)
        with col2:
            option_columns2 = st.selectbox('Chọn biến cần thống kê thứ 2', st.session_state.df.columns)
        option_chart = st.selectbox('Biểu đồ', ['Biểu đồ cột', 'Biểu đồ đường', 'Biểu đồ phân tán'])
        btn = st.button("Thống kê")
        
        if btn:
            st.subheader(f"Biểu đồ cho cặp cột ({option_columns1}, {option_columns2})")
            fig, ax = plt.subplots()

            if option_chart == 'Biểu đồ cột':
                sns.barplot(x=option_columns1, y=option_columns2, data=st.session_state.df, ax=ax)
                ax.set_xlabel(option_columns1)
                ax.set_ylabel(option_columns2)
            elif option_chart == 'Biểu đồ đường':
                sns.lineplot(x=option_columns1, y=option_columns2, data=st.session_state.df, ax=ax)
                ax.set_xlabel(option_columns1)
                ax.set_ylabel(option_columns2)
            elif option_chart == 'Biểu đồ phân tán':
                sns.scatterplot(x=option_columns1, y=option_columns2, data=st.session_state.df, ax=ax)
                ax.set_xlabel(option_columns1)
                ax.set_ylabel(option_columns2)
            st.pyplot(fig)
    else:
        st.warning("Vui lòng nhập dataset")

# Function Matrix
def matrix() :
    if st.session_state.df is not None and not st.session_state.df.empty:
        options2 = st.multiselect('Chọn biến độc lập', st.session_state.columns)
        st.write('Biến độc lập của bạn:', options2)
        if options2:
            st.subheader(f"Biểu đồ ma trận tương quan")
            fig, ax = plt.subplots()
            correlation_matrix = st.session_state.df[options2].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig)

router.add_route('univariate', univariate)
router.add_route('multivariate', multivariate)
router.add_route('matrix', matrix)

if st.sidebar.button('📊 Biểu đồ đơn biến'):
    st.session_state.page = 'univariate'
if st.sidebar.button('📊 Biểu đồ đa biến'):
    st.session_state.page = 'multivariate'
if st.sidebar.button('🔍 Ma trận tương quan'):
    st.session_state.page = 'matrix'

if 'page' not in st.session_state:
    st.session_state.page = 'univariate' 

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