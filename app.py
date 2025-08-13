import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# تحميل الموديلات
best_model = pickle.load(open('best_model.pkl', 'rb'))
nn_model = tf.keras.models.load_model('nn_model.h5')

# تحميل البيانات
df = pd.read_csv('zomato_cleaned.csv')

# إعداد الصفحة
st.set_page_config(page_title="Zomato Analysis & Prediction", layout="wide")

# القائمة الجانبية
menu = st.sidebar.radio("اختر الصفحة:", ["تحليل البيانات", "التنبؤ"])

# --- صفحة التحليل ---
if menu == "تحليل البيانات":
    st.title("تحليل بيانات Zomato")

    st.subheader("عرض أول البيانات")
    st.dataframe(df.head())

    st.subheader("إحصائيات أساسية")
    st.write(df.describe())

    st.subheader("توزيع التقييمات")
    fig, ax = plt.subplots()
    sns.histplot(df['rate'], bins=20, ax=ax, kde=True)
    st.pyplot(fig)

    st.subheader("أكثر المطابخ انتشاراً")
    fig, ax = plt.subplots()
    df['cuisines'].value_counts().head(10).plot(kind='bar', ax=ax)
    st.pyplot(fig)

# --- صفحة التنبؤ ---
elif menu == "التنبؤ":
    st.title("التنبؤ بتصنيف المطعم")

    votes = st.number_input("عدد الأصوات", min_value=0, max_value=20000, value=100)
    rating = st.number_input("التقييم", min_value=0.0, max_value=5.0, value=4.0)

    if st.button("تنبؤ"):
        input_data = np.array([[votes, rating]])

        best_pred = best_model.predict(input_data)
        nn_pred = nn_model.predict(input_data)

        st.success(f"تنبؤ الموديل الأفضل (XGBoost): {best_pred[0]}")
        st.info(f"تنبؤ الشبكة العصبية: {nn_pred[0][0]}")