import pandas as pd
import streamlit as st
import pickle
from sklearn import preprocessing

with open('model.pkl', 'rb') as pickle_model:
    model = pickle.load(pickle_model)

with open('scaler.pkl', 'rb') as pickle_scaler:
    scaler = pickle.load(pickle_scaler)

def main():
    
    st.set_page_config(layout='wide',
                       page_title='Heart Prediction',
                       page_icon='🩺')

    st.title('Heart Failure Prediction')

    st.subheader('Model created to estimate the chance of heart failure.')

    st.markdown('You will need the following data to estimate your chances:')

    variable_dict = {
                    'Age': 'Age of the patient (years)',
                    'Cholesterol':  'Serum cholesterol (mg/dl)',
                    'Max Heart Rate': 'Maximum heart rate achieved (Numeric value between 60 and 202)',
                    'Oldpeak': 'oldpeak = ST (Numeric value measured in depression)',
                    'Chest Pain': 'Chest Pain',
                    'Exercise Angina': 'Exercise-induced angina',
                    'ST Slope (Flat)': 'The slope of the peak exercise ST segment',
                    'ST Slope (Up)': 'The slope of the peak exercise ST segment'  
                    }
    
    variable_table = pd.DataFrame([variable_dict]).T
    variable_table.columns = ['Description']
    st.table(variable_table)

    col1, col2 = st.columns(2)

    with col1:    

        st.header('Do you have this synthoms?')

        if st.checkbox('Chest Pain'):
            chest_pain_type_ASY = 0
        else:
            chest_pain_type_ASY = 1
        
        if st.checkbox('Exercise Angina'):
            exercise_angina = 1
        else:
            exercise_angina = 0

        if st.checkbox('ST Slope: Flat'):
            st_slope_flat = 1
        else:
            st_slope_flat = 0

        if st.checkbox('ST Slope: UP'):
            st_slope_up = 1
        else:
            st_slope_up = 0

    with col2:

        st.header('Fill your data:')

        age = st.number_input('Age: ', min_value=1)
        cholesterol = st.number_input('Cholesterol: ', min_value=1.0, step=.1, format="%.1f")
        max_hr = st.number_input('Max Heart Rate: ', min_value=1)
        oldpeak = st.number_input('Oldpeak: ', min_value=1.0, step=.1, format="%.1f")

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

    input_df = scaler.transform(input_df)

    new_input_df = pd.DataFrame(input_df)
    
    if st.button('Predict'):
        output = model.predict(new_input_df)    

    if output == 1:
        result = 'Yes'
    else:
        result = 'No'

    st.success('Predict: {}'.format(result))

    st.markdown('Developed by: Igor Morais')
    st.write('[Linkedin](https://www.linkedin.com/in/igor-almeida-morais/) │ [GitHub](https://github.com/IgorAMorais)')
    
    
if __name__ == "__main__":
    main()