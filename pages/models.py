import streamlit as st
from sklearn.model_selection import train_test_split
st.set_option('deprecation.showPyplotGlobalUse', False)
import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
import pydotplus
from io import StringIO
from PIL import Image

import model.rnnRegression as rnnRegression
import model.randomForestRegression as randomForestRegression
import model.decisionTreeRegression as decisionTreeRegression
import model.linearRegression as linearRegression
import model.logisticRegression as logisticRegression

import numpy as np
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
st.sidebar.markdown("<h1 style='text-align: center; color: #ff6347;'>Model</h1>", unsafe_allow_html=True)
router = StreamlitRouter()

def linear_regression() :
    st.markdown("<h1 style='color:red;'>Linear Regression</h1>", unsafe_allow_html=True)
    if st.session_state.df is not None and not st.session_state.df.empty:

        option = st.selectbox("Chọn biến mục tiêu", st.session_state.columns, placeholder="Chọn biến mục tiêu")
        st.write('Biến mục tiêu của bạn là:', option)

        options2 = st.multiselect('Chọn biến độc lập', st.session_state.columns)
        st.write('Biến độc lập của bạn:', options2)

        option3 = st.selectbox("Chọn loại biểu đồ",['Biểu đồ đường','Biểu đồ cột','Biểu đồ phân tán','Biểu đồ phân phối lỗi'])
        btn_train = st.button("Train")

        if btn_train :
            y_pred, y_test, accuracy ,model = linearRegression.linearRegression(st.session_state.df, option, options2)
            if option3 == "Biểu đồ đường":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.area_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))
            elif option3 == "Biểu đồ cột":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.line_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))
            elif option3 == "Biểu đồ phân tán":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")   
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=y_test, y=y_pred)

                x_values = [min(y_test), max(y_test)]
                y_values = [min(y_pred), max(y_pred)]
                plt.plot(x_values, y_values, linestyle='--', color='red', label='Linear Regression Model')

                plt.xlabel('Kết quả thực tế')
                plt.ylabel('Dự đoán')
                plt.title('Biểu đồ phân tán giữa kết quả thực tế và dự đoán')
                plt.legend()
                st.pyplot()
            elif option3 == "Biểu đồ hàm mất mát":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                loss = y_pred - y_test
                plt.figure(figsize=(8, 6))
                plt.plot(loss)
                plt.xlabel('Số mẫu')
                plt.ylabel('Hàm mất mát')
                plt.title('Biểu đồ hàm mất mát')
                st.pyplot()
            elif option3 == "Biểu đồ phân phối lỗi":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                plt.figure(figsize=(8, 6))
                sns.histplot(y_pred - y_test, kde=True)
                plt.xlabel('Lỗi (Dự đoán - Thực tế)')
                plt.ylabel('Tần suất')
                plt.title('Biểu đồ phân phối lỗi')
                st.pyplot()

def logistic_regression() :
    st.markdown("<h1 style='color:red;'>Logistic Regression</h1>", unsafe_allow_html=True)
    if st.session_state.df is not None and not st.session_state.df.empty:

        option = st.selectbox("Chọn biến mục tiêu", st.session_state.columns, placeholder="Chọn biến mục tiêu")
        st.write('Biến mục tiêu của bạn là:', option)
        options2 = st.multiselect('Chọn biến độc lập', st.session_state.columns)
        st.write('Biến độc lập của bạn:', options2)
        option3 = st.selectbox("Chọn loại biểu đồ",['Biểu đồ đường','Biểu đồ cột','Biểu đồ ma trận','Biểu đồ phân tán'])
        btn_train = st.button("Train")

        if btn_train :
            X_train, X_test, y_pred, y_test, model, accuracy = logisticRegression.logisticRegression(st.session_state.df, option, options2)
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()

            if option3 == "Biểu đồ đường":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.area_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))

            elif option3 == "Biểu đồ cột":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.line_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))
            elif option3 == "Biểu đồ ma trận":
                plt.figure(figsize=(10, 6))
                sns.heatmap(report_df.iloc[:-1, :].astype(float), annot=True, cmap='coolwarm')
                plt.title('Classification Report Heatmap')
                st.pyplot()

def cnn_regression() :
    st.markdown("<h1 style='color:red;'>RNN Regression</h1>", unsafe_allow_html=True)
    if st.session_state.df is not None and not st.session_state.df.empty:

        option = st.selectbox("Chọn biến mục tiêu", st.session_state.columns, placeholder="Chọn biến mục tiêu")
        st.write('Biến mục tiêu của bạn là:', option)

        options2 = st.multiselect('Chọn biến độc lập', st.session_state.columns)
        st.write('Biến độc lập của bạn:', options2)

        option3 = st.selectbox("Chọn loại biểu đồ",['Biểu đồ đường','Biểu đồ cột','Biểu đồ phân tán','Biểu đồ phân phối lỗi'])
        btn_train = st.button("Train")

        if btn_train :
            y_pred, y_test, accuracy = rnnRegression.cnn_regression(st.session_state.df, option, options2)
            if option3 == "Biểu đồ đường":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.area_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))
            elif option3 == "Biểu đồ cột":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.line_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))
            elif option3 == "Biểu đồ phân tán":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=y_test, y=y_pred)

                x_values = [min(y_test), max(y_test)]
                y_values = [min(y_pred), max(y_pred)]
                plt.plot(x_values, y_values, linestyle='--', color='red', label='Linear Regression Model')

                plt.xlabel('Kết quả thực tế')
                plt.ylabel('Dự đoán')
                plt.title('Biểu đồ phân tán giữa kết quả thực tế và dự đoán')
                plt.legend()
                st.pyplot()
            elif option3 == "Biểu đồ phân phối lỗi":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                plt.figure(figsize=(8, 6))
                sns.histplot(y_pred - y_test, kde=True)
                plt.xlabel('Lỗi (Dự đoán - Thực tế)')
                plt.ylabel('Tần suất')
                plt.title('Biểu đồ phân phối lỗi')
                st.pyplot()

def randomForest_Regression() :
    st.markdown("<h1 style='color:red;'>RandomForest Regression</h1>", unsafe_allow_html=True)
    if st.session_state.df is not None and not st.session_state.df.empty:

        option = st.selectbox("Chọn biến mục tiêu", st.session_state.columns, placeholder="Chọn biến mục tiêu")
        st.write('Biến mục tiêu của bạn là:', option)

        options2 = st.multiselect('Chọn biến độc lập', st.session_state.columns)
        st.write('Biến độc lập của bạn:', options2)

        option3 = st.selectbox("Chọn loại biểu đồ",['Biểu đồ đường','Biểu đồ cột','Biểu đồ phân tán','Biểu đồ phân phối lỗi'])
        btn_train = st.button("Train")

        if btn_train :
            y_pred, y_test, accuracy = randomForestRegression.randomForestRegression(st.session_state.df, option, options2)
            if option3 == "Biểu đồ cột":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.area_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))
            elif option3 == "Biểu đồ đường":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.line_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))
            elif option3 == "Biểu đồ phân tán":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=y_test, y=y_pred)

                x_values = [min(y_test), max(y_test)]
                y_values = [min(y_pred), max(y_pred)]
                plt.plot(x_values, y_values, linestyle='--', color='red', label='Linear Regression Model')

                plt.xlabel('Kết quả thực tế')
                plt.ylabel('Dự đoán')
                plt.title('Biểu đồ phân tán giữa kết quả thực tế và dự đoán')
                plt.legend()
                st.pyplot()
            elif option3 == "Biểu đồ phân phối lỗi":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                plt.figure(figsize=(8, 6))
                sns.histplot(y_pred - y_test, kde=True)
                plt.xlabel('Lỗi (Dự đoán - Thực tế)')
                plt.ylabel('Tần suất')
                plt.title('Biểu đồ phân phối lỗi')
                st.pyplot()

def decisionTree_Regression() :
    st.markdown("<h1 style='color:red;'>DecisionTree Regression</h1>", unsafe_allow_html=True)
    if st.session_state.df is not None and not st.session_state.df.empty:

        option = st.selectbox("Chọn biến mục tiêu", st.session_state.columns, placeholder="Chọn biến mục tiêu")
        st.write('Biến mục tiêu của bạn là:', option)

        options2 = st.multiselect('Chọn biến độc lập', st.session_state.columns)
        st.write('Biến độc lập của bạn:', options2)

        option3 = st.selectbox("Chọn loại biểu đồ",['Biểu đồ hình cây','Biểu đồ đường','Biểu đồ cột','Biểu đồ phân tán','Biểu đồ phân phối lỗi'])
        btn_train = st.button("Train")

        if btn_train :
            model , y_pred, y_test, accuracy = decisionTreeRegression.decisionTreeRegression(st.session_state.df, option, options2)
            if option3 == "Biểu đồ cột":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.area_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))
            elif option3 == "Biểu đồ đường":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.line_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))
            elif option3 == "Biểu đồ phân tán":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=y_test, y=y_pred)
                plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], linestyle='--', color='red')
                plt.xlabel('Kết quả thực tế')
                plt.ylabel('Dự đoán')
                plt.title('Biểu đồ phân tán giữa kết quả thực tế và dự đoán')
                st.pyplot()
            elif option3 == "Biểu đồ phân phối lỗi":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                plt.figure(figsize=(8, 6))
                sns.histplot(y_pred - y_test, kde=True)
                plt.xlabel('Lỗi (Dự đoán - Thực tế)')
                plt.ylabel('Tần suất')
                plt.title('Biểu đồ phân phối lỗi')
                st.pyplot()
            elif option3 == "Biểu đồ hình cây":
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.write("Biểu đồ hình cây của DecisionTree")
                plt.figure(figsize=(20, 10))
                plot_tree(model, feature_names=options2, filled=True, rounded=True, fontsize=10)
                st.pyplot()

router.add_route('linear_regression', linear_regression)
router.add_route('logistic_regression', logistic_regression)
router.add_route('cnn_regression', cnn_regression)
router.add_route('randomForest_Regression', randomForest_Regression)
router.add_route('decisionTree_Regression', decisionTree_Regression)

if st.sidebar.button('🔍 DecisionTree Regression'):
    st.session_state.page = 'decisionTree_Regression'
if st.sidebar.button('🔍 Linear Regression'):
    st.session_state.page = 'linear_regression'
if st.sidebar.button('🔍 Logistic Regression'):
    st.session_state.page = 'logistic_regression'
if st.sidebar.button('🔍 RWNN Regression'):
    st.session_state.page = 'rnn_regression'
if st.sidebar.button('🔍 RandomForest Regression'):
    st.session_state.page = 'randomForest_Regression'
if 'page' not in st.session_state:
    st.session_state.page = 'linear_regression' 

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