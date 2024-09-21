import streamlit as st
from pydantic import BaseModel
import pandas as pd
import pickle
import tempfile  # Biblioteca para crear archivos temporales
import shutil 
from typing import Optional
from typing import ClassVar
from sklearn.linear_model import Ridge
from pycaret.regression import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model
from pycaret.classification import predict_model

# Cargar el modelo preentrenado desde el archivo pickle
#model_path = "best_model.pkl"
with open("modelo_ridge.pkl", 'rb') as model_file:
    dt2 = pickle.load(model_file)

#if 'test_data' not in st.session_state:
 #   st.session_state['test_data'] = pd.read_csv('prueba_APP.csv')
#prueba = pd.read_csv("prueba_APP.csv",header = 0,sep=";",decimal=",")

def prediccion_individual():
    st.header("API de Predicción Precio")

    # Entradas del usuario para los selectbox en el orden de la imagen
    Email = st.text_input("Email", value="amaris@usantoto.com")
    Address = st.selectbox("Address", ['Munich', 'Ausburgo', 'Berlin', 'Frankfurt'], index=0)
    dominio = st.selectbox("dominio", ['yahoo', 'Otro', 'gmail', 'hotmail'], index=0)
    Tec = st.selectbox("Tec", ['PC', 'Smartphone', 'Iphone', 'Portatil'], index=0)
    Avg_Session_Length = st.number_input("Avg Session Length", value=33.946241)
    Time_on_App = st.number_input("Time on App", value=10.983977)
    Time_on_Website = st.number_input("Time on Website", value=37.951489)
    Length_of_Membership = st.number_input("Length of Membership", value=3.050713)

    # Convertir los valores de texto a números si es posible
    if st.button("Calcular"):
        try: 
            prueba = pd.read_csv("prueba_APP.csv",header = 0,sep=";",decimal=",")

            # Crear el dataframe a partir de los inputs del usuario
            user = pd.DataFrame({
                'x0':[Email],'x1':[Address],'x2':[dominio],'x3': [Tec],
                'x4': [Avg_Session_Length], 'x5': [Time_on_App], 'x6': [Time_on_Website], 'x7':[Length_of_Membership], 'x8':[0]
            })

            # Asegurar que las columnas coincidan con las del dataset de prueba
            user.columns = prueba.columns

            # Concatenar los datos del usuario con los datos de prueba
            prueba2 = pd.concat([user,prueba],axis = 0)
            prueba2.index = range(prueba2.shape[0])

            # Hacer predicciones
            predictions = predict_model(dt2, data=prueba2)

            st.write(f'La predicción es: {predictions.iloc[0]["prediction_label"]}')

        except ValueError:
            st.error("Por favor, ingrese valores numéricos válidos en los campos correspondientes.")


    if st.button("Volver al menú principal"):
        st.session_state['menu'] = 'main'


# Función para predicción por base de datos
def prediccion_base_datos():
    # Título de la API
    st.title("API de Predicción precio")

    # Botón para subir archivo Excel
    uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx", "csv"])

    # Botón para predecir
    if st.button("Predecir"):
        if uploaded_file is not None:
            try:
                # Cargar el archivo subido
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                if uploaded_file.name.endswith(".csv"):
                    prueba = pd.read_csv(tmp_path,header = 0,sep=";",decimal=",")
                else:
                    prueba = pd.read_excel(tmp_path)

                df_test = prueba.copy()
                predictions = predict_model(dt2, data=df_test)
                predictions["price"] = predictions["prediction_label"]

                # Preparar archivo para descargar
                kaggle = pd.DataFrame({'Email': prueba["Email"], 'price': predictions["price"]})

                # Mostrar predicciones en pantalla
                st.write("Predicciones generadas correctamente!")
                st.write(kaggle)

                # Botón para descargar el archivo de predicciones
                st.download_button(label="Descargar archivo de predicciones",
                                data=kaggle.to_csv(index=False),
                                file_name="kaggle_predictions.csv",
                                mime="text/csv")

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Por favor, cargue un archivo válido.")

    if st.button("Volver al menú principal"):
        st.session_state['menu'] = 'main'



# Función principal para mostrar el menú de opciones
def menu_principal():
    st.title("API de Predicción Precio")
    option = st.selectbox("Seleccione una opción", ["", "Predicción Individual", "Predicción Base de Datos"])

    if option == "Predicción Individual":
        st.session_state['menu'] = 'individual'
    elif option == "Predicción Base de Datos":
        st.session_state['menu'] = 'base_datos'

# Lógica para manejar el flujo de la aplicación
if 'menu' not in st.session_state:
    st.session_state['menu'] = 'main'

if st.session_state['menu'] == 'main':
    menu_principal()
elif st.session_state['menu'] == 'individual':
    prediccion_individual()
elif st.session_state['menu'] == 'base_datos':
    prediccion_base_datos()



