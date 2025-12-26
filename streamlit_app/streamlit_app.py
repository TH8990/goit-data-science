import streamlit as st
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

# Налаштування сторінки
st.set_page_config(
    page_title="Fashion-MNIST Classifier",
    page_icon="👕",
    layout="centered"
)

# Словник класів Fashion-MNIST
CLASS_NAMES = [
    'Футболка/топ', 'Штани', 'Пуловер', 'Сукня', 'Пальто',
    'Сандалі', 'Сорочка', 'Кросівки', 'Сумка', 'Черевики'
]

def load_resources():
    """Завантаження історії навчання та моделі"""
    history = None
    model = None
    
    try:
        with open('cnn_history.json', 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        pass
        
    try:
        # Завантажуємо модель Fashion-MNIST
        model = load_model('base_cnn_model.h5')
    except Exception:
        pass
        
    return history, model

def main():
    st.title("👗 Fashion-MNIST CNN Classifier")
    st.markdown("""
    Цей застосунок класифікує елементи одягу за допомогою нейронної мережі, 
    навченої на датасеті **Fashion-MNIST**.
    """)

    history, model = load_resources()

    tab1, tab2 = st.tabs(["📊 Графіки навчання", "🔍 Розпізнавання одягу"])

    with tab1:
        st.header("Аналіз метрик")
        if history:
            df = pd.DataFrame(history)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Точність (Accuracy)")
                st.line_chart(df[['accuracy', 'val_accuracy']])
            
            with col2:
                st.subheader("Втрати (Loss)")
                st.line_chart(df[['loss', 'val_loss']])
            
            st.info(f"Найкраща точність на валідації: {max(history['val_accuracy']):.2%}")
        else:
            st.warning("Файл 'cnn_history.json' не знайдено.")

    with tab2:
        st.header("Завантажте фото одягу")
        if model:
            uploaded_file = st.file_uploader("Оберіть зображення...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                img = Image.open(uploaded_file)
                
                # Відображення оригіналу
                st.image(img, caption='Ваше зображення', width=300)
                
                # Передобробка для Fashion-MNIST:
                # 1. Перетворення в Grayscale (чорно-біле)
                # 2. Зміна розміру на 28x28
                # 3. Інверсія кольорів (якщо фон світлий, бо модель вчилася на чорному фоні)
                
                img_gray = ImageOps.grayscale(img)
                img_resized = img_gray.resize((28, 28))
                
                # Перетворення в масив та нормалізація
                img_array = image.img_to_array(img_resized)
                img_array = img_array.reshape(1, 28, 28, 1)
                img_array = img_array.astype('float32') / 255.0
                
                if st.button('Класифікувати'):
                    with st.spinner('Обробка...'):
                        predictions = model.predict(img_array)
                        result_index = np.argmax(predictions[0])
                        confidence = predictions[0][result_index]
                    
                    st.success(f"Це **{CLASS_NAMES[result_index]}** з імовірністю {confidence:.2%}")
                    
                    # Гістограма всіх класів
                    chart_data = pd.DataFrame({
                        'Категорія': CLASS_NAMES,
                        'Ймовірність': predictions[0]
                    }).sort_values(by='Ймовірність', ascending=False)
                    
                    st.bar_chart(chart_data.set_index('Категорія'))
        else:
            st.error("Модель 'base_cnn_model.h5' не знайдено.")

if __name__ == "__main__":
    main()