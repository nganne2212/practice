import streamlit as st
import pickle
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from datetime import datetime

# Cấu hình trang
st.set_page_config(page_title="🌸 Iris Flower Classifier", layout="wide", initial_sidebar_state="expanded")

iris = load_iris()

# Load the trained model and scaler
clf = pickle.load(open('iris_model.pkl', 'rb'))
scaler = pickle.load(open('iris_scaler.pkl', 'rb'))

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Dữ liệu về các loài hoa
iris_info = {
    'setosa': '🌼 Loài Setosa - Hoa nhỏ, cánh hoa tròn',
    'versicolor': '🌺 Loài Versicolor - Hoa vừa, cánh hoa dài',
    'virginica': '🌹 Loài Virginica - Hoa lớn, cánh hoa rất dài'
}

iris_chars = {
    'setosa': {'sepal_length': '4.3-5.8', 'sepal_width': '2.3-4.4', 'petal_length': '1.0-1.9', 'petal_width': '0.1-0.6'},
    'versicolor': {'sepal_length': '5.5-7.0', 'sepal_width': '2.0-3.4', 'petal_length': '3.0-5.1', 'petal_width': '1.0-1.8'},
    'virginica': {'sepal_length': '6.3-7.9', 'sepal_width': '2.2-3.8', 'petal_length': '4.9-6.9', 'petal_width': '1.4-2.5'}
}

# Tiêu đề chính
st.title("🌸 Iris Flower Classifier")
st.markdown("**Ứng dụng dự đoán loại hoa Iris bằng Machine Learning (Random Forest)**")
st.markdown("---")

# Tạo tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Dự Đoán", "📊 Thống Kê", "📁 Batch Prediction", "📈 Thông Tin Mô Hình"])

# ===== TAB 1: DỰ ĐOÁN =====
with tab1:
    col1, col2 = st.columns(2)
    
    # Cột 1: Nhập dữ liệu
    with col1:
        st.subheader("📝 Nhập Đặc Tính Hoa")
        sepal_length = st.slider('Chiều dài đài hoa (cm)', 4.0, 8.0, 5.8, 0.1)
        sepal_width = st.slider('Chiều rộng đài hoa (cm)', 2.0, 4.5, 3.0, 0.1)
        petal_length = st.slider('Chiều dài cánh hoa (cm)', 1.0, 7.0, 4.0, 0.1)
        petal_width = st.slider('Chiều rộng cánh hoa (cm)', 0.1, 2.5, 1.3, 0.1)
        
        predict_btn = st.button("🔮 Dự Đoán", key="predict_btn", use_container_width=True)
    
    # Cột 2: Hiển thị kết quả
    with col2:
        st.subheader("📊 Kết Quả Dự Đoán")
        
        if predict_btn:
            # Make predictions
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = scaler.transform(input_data)
            prediction = clf.predict(input_scaled)[0]
            probabilities = clf.predict_proba(input_scaled)[0]
            
            # Hiển thị loài hoa được dự đoán
            predicted_species = iris.target_names[prediction]
            st.success(f"### {iris_info[predicted_species]}")
            
            # Hiển thị xác suất
            st.markdown("#### 📈 Xác Suất Dự Đoán:")
            for i, prob in enumerate(probabilities):
                species = iris.target_names[i]
                st.write(f"{iris_info[species]}")
                st.progress(prob)
                st.write(f"Xác suất: **{prob*100:.2f}%**")
                st.write("")
            
            # Lưu vào lịch sử
            st.session_state.prediction_history.append({
                'Thời gian': datetime.now().strftime("%H:%M:%S"),
                'Đài (dài)': sepal_length,
                'Đài (rộng)': sepal_width,
                'Cánh (dài)': petal_length,
                'Cánh (rộng)': petal_width,
                'Loài': iris_info[predicted_species],
                'Độ tin cậy': f"{max(probabilities)*100:.2f}%"
            })
    
    # Thông tin chi tiết
    st.markdown("---")
    st.subheader("📋 Thông Tin Đặc Tính Đã Nhập")
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.metric("Đài hoa (dài)", f"{sepal_length:.2f} cm")
    with col_info2:
        st.metric("Đài hoa (rộng)", f"{sepal_width:.2f} cm")
    with col_info3:
        st.metric("Cánh hoa (dài)", f"{petal_length:.2f} cm")
    with col_info4:
        st.metric("Cánh hoa (rộng)", f"{petal_width:.2f} cm")


# ===== TAB 2: THỐNG KÊ =====
with tab2:
    st.subheader("📊 Thống Kê và Phân Tích")
    
    if st.session_state.prediction_history:
        df_history = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tổng dự đoán", len(df_history))
        with col2:
            st.metric("Loài phổ biến nhất", 
                     df_history['Loài'].value_counts().index[0] if len(df_history) > 0 else "N/A")
        with col3:
            st.metric("Độ tin cậy trung bình", 
                     f"{df_history['Độ tin cậy'].str.rstrip('%').astype(float).mean():.2f}%")
        
        st.markdown("---")
        st.subheader("📈 Bảng Lịch Sử Dự Đoán")
        st.dataframe(df_history, use_container_width=True)
        
        # Nút xóa lịch sử
        if st.button("🗑️ Xóa lịch sử"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("⚠️ Chưa có dự đoán nào. Vui lòng dự đoán ở tab 'Dự Đoán'")


# ===== TAB 3: BATCH PREDICTION =====
with tab3:
    st.subheader("📁 Batch Prediction (Dự Đoán Hàng Loạt)")
    
    uploaded_file = st.file_uploader("Chọn file CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📄 Dữ liệu tải lên:")
            st.dataframe(df, use_container_width=True)
            
            if st.button("🔮 Dự Đoán Hàng Loạt"):
                # Chuẩn hóa và dự đoán
                X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values
                X_scaled = scaler.transform(X)
                predictions = clf.predict(X_scaled)
                probabilities = clf.predict_proba(X_scaled)
                
                # Tạo bảng kết quả
                df_results = df.copy()
                df_results['Dự Đoán'] = [iris.target_names[p] for p in predictions]
                df_results['Độ Tin Cậy'] = [f"{max(prob)*100:.2f}%" for prob in probabilities]
                
                st.write("✅ Kết Quả Dự Đoán:")
                st.dataframe(df_results, use_container_width=True)
                
                # Tải xuống kết quả
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="⬇️ Tải xuống kết quả (CSV)",
                    data=csv,
                    file_name="iris_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
    else:
        st.info("💡 Mẹo: File CSV cần có các cột: 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'")


# ===== TAB 4: THÔNG TIN MÔ HÌNH =====
with tab4:
    st.subheader("📈 Thông Tin Mô Hình")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Loại Mô Hình", "Random Forest")
    with col2:
        st.metric("Số Cây", "200")
    with col3:
        st.metric("Độ Sâu Tối Đa", "15")
    
    st.markdown("---")
    st.subheader("🎯 Dữ Liệu Huấn Luyện")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng Mẫu", "150")
    with col2:
        st.metric("Mẫu Huấn Luyện", "120")
    with col3:
        st.metric("Mẫu Kiểm Tra", "30")
    with col4:
        st.metric("Số Đặc Tính", "4")
    
    st.markdown("---")
    st.subheader("🌸 Các Loài Hoa (Target Classes)")
    for species in iris.target_names:
        st.write(iris_info[species])
        st.write(f"**Đặc tính điển hình:**")
        chars = iris_chars[species]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption(f"Đài (dài): {chars['sepal_length']} cm")
        with col2:
            st.caption(f"Đài (rộng): {chars['sepal_width']} cm")
        with col3:
            st.caption(f"Cánh (dài): {chars['petal_length']} cm")
        with col4:
            st.caption(f"Cánh (rộng): {chars['petal_width']} cm")
        st.write("")