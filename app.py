import pandas as pd
import streamlit as st
from joblib import dump, load
from sklearn import preprocessing

model = load("model.pkl")
scaler = load("scaler.pkl")

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

    output=''

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

    #input_df = scaler.fit(input_df)

    #new_input_df = pd.DataFrame([input_df])

    st.dataframe(scaler.feature_names_in_)

    if st.button('Predict'):
        output = model.predict(new_input_df)    

    if output == 1:
        result = 'Yes'
    else:
        result = 'No'

    st.success('Predict: {}'.format(result))

if __name__ == "__main__":
    main()
