import streamlit as st
import joblib
import pandas as pd

# โหลดโมเดลและคอลัมน์ที่เซฟไว้
best_model = joblib.load('best_ann_model.pkl')
columns = joblib.load('df_columns.pkl')

# แสดงชื่อแอป
st.title("🎈 Diamond Price Prediction App")
st.write("Let's predict the price of diamonds!")

# สร้าง input fields สำหรับการป้อนค่าของคุณสมบัติเพชร
carat = st.number_input("Carat Weight", min_value=0.01, step=0.01)
cut = st.selectbox("Cut", options=["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", options=["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity", options=["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])
depth = st.number_input("Depth Percentage", min_value=40.0, max_value=80.0, step=0.1)
table = st.number_input("Table Percentage", min_value=40.0, max_value=100.0, step=0.1)
x = st.number_input("Length (mm)", min_value=0.2, step=0.1)
y = st.number_input("Width (mm)", min_value=0.2, step=0.1)
z = st.number_input("Height (mm)", min_value=0.2, step=0.1)

# สร้างดิกชันนารีของค่าที่ป้อน
input_data = {
    "carat": carat,
    "cut": cut,
    "color": color,
    "clarity": clarity,
    "depth": depth,
    "table": table,
    "x": x,
    "y": y,
    "z": z,
}

# แปลง input เป็น DataFrame
input_df = pd.DataFrame([input_data])

# ทำ One-Hot Encoding กับข้อมูล input
input_df_encoded = pd.get_dummies(input_df)

# เติมคอลัมน์ที่ขาดไปด้วย 0 เพื่อให้ฟีเจอร์ที่ใช้งานตรงกับตอนฝึกโมเดล
for col in columns:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

# จัดเรียงคอลัมน์ให้ตรงกับโมเดลที่ใช้
input_df_encoded = input_df_encoded[columns]

# เมื่อกดปุ่ม 'Predict' จะทำนายราคาของเพชร
if st.button("Predict"):
    # ทำนายราคาเพชรโดยใช้โมเดลที่โหลดมา
    prediction = best_model.predict(input_df_encoded)

    # คูณค่าการพยากรณ์ด้วย 100,000
    predicted_price = prediction[0] * 100000 * -1

    # แสดงผลการทำนาย
    st.write(f"The predicted price of the diamond is: ${predicted_price:,.2f}")
