import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model=pickle.load(open('trained_model.sav','rb'))


def medical_prediction(input_data):
    

    #changing input_data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data,dtype=np.float32)

    #reshaping the array
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print("The inasurence cost in USD")
    return prediction;
    
  
def main():
    #giving a title
    st.title('Medical Cost Prediction Web App')
    
    #getting the inout data from the user
    
    age=st.text_input('Age of person:')
    
    sex=st.text_input('Gender of person[Male=0 ,Female=1]:')
    
    bmi=st.text_input('BMI of person:')
    
    children=st.text_input('children of person:')
    
    smoker=st.text_input('Smoker or not:[Yes=0,No=1]')
    
    region=st.text_input("Region of person:['Southeast':0, 'Southwest':1 ,'Northeast':2, 'Northwest':3]")
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Cost Test Result'):
        diagnosis = medical_prediction([float(age),float(sex),float(children),float(bmi),float(smoker),float(region)])
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
