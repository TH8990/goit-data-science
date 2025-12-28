import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model

# Налаштування сторінки
st.set_page_config(page_title="Fashion-MNIST CNN Explorer", layout="wide")

# Визначаємо шляхи до файлів проекту
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "base_cnn_model.h5")
HISTORY_PATH = os.path.join(BASE_DIR, "cnn_history.json")

@st.cache_resource
def load_trained_model():
    """Завантаження збереженої моделі CNN."""
    if os.path.exists(MODEL_PATH):
        try:
            # Завантажуємо модель
            return load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Не вдалося завантажити модель: {e}")
            return None
    return None

@st.cache_data
def load_project_data():
    """Завантаження даних Fashion-MNIST та історії навчання."""
    (_, _), (x_test, y_test) = fashion_mnist.load_data()
    
    history = None
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, 'r') as f:
                history = json.load(f)
        except Exception:
            pass
            
    return x_test, y_test, history

def main():
    st.title("👕 Класифікація одягу за допомогою CNN")
    st.markdown("""
    Цей додаток демонструє роботу базової архітектури **CNN**
    """)

    # Завантаження ресурсів
    model = load_trained_model()
    x_test, y_test, history = load_project_data()

    class_names = [
        'Футболка/топ', 'Труси/штани', 'Пуловер', 'Сукня', 'Пальто',
        'Сандалі', 'Сорочка', 'Кросівки', 'Сумка', 'Черевики'
    ]

    # Створення вкладок
    tab1, tab2 = st.tabs(["🚀 Тестування моделі", "📊 Метрики та Аналіз"])

    with tab1:
        st.header("Перевірка на випадкових зразках")
        
        # Відображення загальної точності зверху для швидкого ознайомлення
        st.info("**Підсумкова точність моделі на тестовій вибірці становить: 93.14%**")
        
        if st.button("🎲 Обрати випадковий зразок для тесту"):
            idx = np.random.randint(0, len(x_test))
            st.session_state['sample_img'] = x_test[idx]
            st.session_state['sample_label'] = y_test[idx]

        if 'sample_img' in st.session_state:
            col1, col2 = st.columns([1, 1])
            
            img = st.session_state['sample_img']
            true_label = class_names[st.session_state['sample_label']]

            with col1:
                st.write("**Вхідне зображення (Fashion-MNIST):**")
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
                st.write(f"Очікуваний клас: **{true_label}**")

            with col2:
                if model:
                    # Підготовка: (1, 28, 28, 1) та нормалізація
                    input_img = img.astype('float32').reshape(1, 28, 28, 1) / 255.0
                    preds = model.predict(input_img)
                    pred_idx = np.argmax(preds[0])
                    confidence = preds[0][pred_idx]

                    color = "green" if pred_idx == st.session_state['sample_label'] else "red"
                    st.markdown(f"### Прогноз моделі: <span style='color:{color}'>{class_names[pred_idx]}</span>", unsafe_allow_html=True)
                    st.metric("Впевненість (Confidence)", f"{confidence*100:.2f}%")
                    
                    # Візуалізація розподілу ймовірностей
                    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
                    y_pos = np.arange(len(class_names))
                    ax_bar.barh(y_pos, preds[0], align='center', color='teal')
                    ax_bar.set_yticks(y_pos)
                    ax_bar.set_yticklabels(class_names)
                    ax_bar.invert_yaxis()
                    ax_bar.set_xlabel('Ймовірність')
                    ax_bar.set_title('Розподіл по класах')
                    st.pyplot(fig_bar)
                else:
                    st.error("Файл моделі `base_cnn_model.h5` не знайдено.")

    with tab2:
        st.header("Статистика навчання та тестування")
        
        # Цифрове відображення точності
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Тестова точність (Accuracy)", "93.14%")
        with c2:
            st.metric("Кількість класів", "10")
        with c3:
            st.metric("Розмір входу", "28x28x1")

        if history:
            col_acc, col_loss = st.columns(2)
            
            with col_acc:
                st.subheader("Графік Accuracy")
                fig_acc, ax_acc = plt.subplots()
                ax_acc.plot(history['accuracy'], label='Тренування', color='blue')
                ax_acc.plot(history['val_accuracy'], label='Валідація', color='orange')
                ax_acc.set_title('Зміна точності по епохах')
                ax_acc.set_xlabel('Епоха')
                ax_acc.set_ylabel('Точність')
                ax_acc.legend()
                ax_acc.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig_acc)

            with col_loss:
                st.subheader("Графік Loss")
                fig_loss, ax_loss = plt.subplots()
                ax_loss.plot(history['loss'], label='Тренування', color='blue')
                ax_loss.plot(history['val_loss'], label='Валідація', color='orange')
                ax_loss.set_title('Зміна втрат по епохах')
                ax_loss.set_xlabel('Епоха')
                ax_loss.set_ylabel('Втрати')
                ax_loss.legend()
                ax_loss.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig_loss)
        else:
            st.warning("Дані історії (`cnn_history.json`) відсутні. Відображення графіків неможливе.")

if __name__ == "__main__":
    main()