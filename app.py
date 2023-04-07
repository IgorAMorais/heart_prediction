import pandas as pd
import streamlit as st
from joblib import dump, load

model = load("model.pkl")

def main():
    #t.sidebar.image() #Colocar imagem relacionada ao projeto aqui
    #st.sidebar.markdown() #Colocar link para o linkedin aqui
    st.title('Heart Failure Prediction')

    #st.markdown() #Colocar quais variáveis estão sendo levadas em consideração na modelagem

    age = st.number_input('Age: ', min_value=1)
    cholesterol = st.number_input('Cholesterol: ', min_value=1)
    max_hr = st.number_input('MaxHR: ', min_value=1)
    oldpeak = st.number_input('Oldpeak: ', min_value=1)
    
    if st.checkbox('Chest Pain Type'):
       chest_pain_type_ASY = 0
    else:
        chest_pain_type_ASY = 1
     
    exercise_angina = st.checkbox('Exercise Angina')
    st_slope_flat = st.checkbox('ST Slope: Flat')
    st_slope_up = st.checkbox('ST Slope: UP')
    teste = st.checkbox('ST Slope?')

    input_dict = {
                'Age': age,
                'Cholesterol': cholesterol,
                'MaxHR': max_hr,
                'Oldpeak': oldpeak,
                'ChestPainType_ASY': chest_pain_type_ASY,
                'ExerciseAngina_Y': exercise_angina,
                'ST_Slope_Flat': st_slope_flat,
                'ST_Slope_Up': st_slope_up
                }
    
    input_df = pd.DataFrame([input_dict])

    if st.button('Predict'):
        output = model.predict(input_df)

""" if output == 1:
        result = 'Yes'
    else:
        result = 'No'
""" 
st.success('Prediction: {output}')

if __name__ == "__main__":
    main()
