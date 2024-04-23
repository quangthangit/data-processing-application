import streamlit as st
import pandas as pd
import model.logistic as logistic
import model.regression as regression
import model.knn_regression as knn_regression
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

if 'file' not in st.session_state:
    st.session_state.file = None
if 'columns' not in st.session_state:
    st.session_state.columns = []
if 'df' not in st.session_state:
    st.session_state.df = None

select_file = st.file_uploader("SELECT FILE")
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


if st.session_state.columns:

    btn_checkNull = st.button("Kiểm tra dữ liệu")
    if btn_checkNull : 
        st.write(st.session_state.df.isnull().sum())

    option_columns = st.selectbox('Chọn dữ liệu cần sử lý',st.session_state.columns )
    option_data = st.selectbox('Chọn cách xử lý',['Xóa dữ liệu null','Lấy giá trị trung bình','Lấy theo tuần suất xuất hiện'] )
    btn_cleanNull = st.button("Xử lý dữ liệu")

    option_chart = st.selectbox('Biểu đồ',['1 Biến phân loại','1 Biến định lượng','2 Biến phân loại', '1 Biến phân loại 1 biến định lượng','2 Biến định lượng'])
    option_columns_chart =  st.selectbox('Chọn biến',st.session_state.columns )
    btn_chart = st.button("Thống kê")

    if btn_chart:
        if option_chart == "1 Biến phân loại":
            st.write('Biểu đồ cột thống kê tần suất:')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(data=st.session_state.df, x=option_columns_chart, ax=ax)
            st.pyplot(fig)
            st.write('Biểu đồ cột thống kê tần suất:')

    if btn_cleanNull:
        if option_data == 'Xóa dữ liệu null':
            st.session_state.df = st.session_state.df.dropna(subset=[option_columns])
            st.write(st.session_state.df)
        elif option_data == "Lấy giá trị trung bình" : 
            mean_values = st.session_state.df[option_columns].mean()
            st.session_state.df[option_columns].fillna(mean_values, inplace=True)
            st.write(st.session_state.df.isnull().sum())
        else :
            most_frequent_value = st.session_state.df[option_columns].mode()[0]
            st.session_state.df[option_columns].fillna(most_frequent_value, inplace=True)
            st.write(st.session_state.df.isnull().sum())

    option = st.selectbox("Chọn biến mục tiêu", st.session_state.columns, placeholder="Chọn biến mục tiêu")
    st.write('Biến mục tiêu của bạn là:', option)

    options2 = st.multiselect('Chọn biến độc lập', st.session_state.columns)
    st.write('Biến độc lập của bạn:', options2)

    options3 = st.selectbox('Chọn mô hình', ['Logistic', 'Regression' , 'KNN'])

    btn_train = st.button("Train")

    option4 = st.selectbox("Chọn loại biểu đồ",['Biểu đồ đường','Biểu đồ cột'])

    if btn_train:
        if option4 == "Biểu đồ đường" : 
            if options3 == "Logistic" :
                y_pred, y_test, accuracy = logistic.logistic(st.session_state.df, option, options2)
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.area_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))
            elif options3 == "Regression" :
                y_pred, y_test, accuracy = regression.linearRegression(st.session_state.df, option, options2)
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.area_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))
            else :
                y_pred, y_test, accuracy = knn_regression.knn_regression(st.session_state.df, option, options2)
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.area_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))
        else :
            if options3 == "Logistic" :
                y_pred, y_test, accuracy = logistic.logistic(st.session_state.df, option, options2)
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.line_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))
            elif options3 == "Regression" :
                y_pred, y_test, accuracy = regression.linearRegression(st.session_state.df, option, options2)
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.line_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))
            else :
                y_pred, y_test, accuracy = knn_regression.knn_regression(st.session_state.df, option, options2)
                st.success(f"Độ chính xác của mô hình là: {accuracy * 100:.2f}%")
                st.line_chart(pd.DataFrame({'Dự đoán': y_pred, 'Kết quả thực tế': y_test.values}))

    