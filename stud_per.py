import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler,LabelEncoder

le=LabelEncoder()  # issue with this
def load_model():
    with open("student_lr_final_model.pkl",'rb') as file:
        model,scaler,le=pickle.load(file)
    #print("Loaded Label Encoder Classes:",le.classes_)
    return model,scaler,le
def preprocessing_input_data(data,scaler,le):
    data["Extracurricular Activities"]=le.fit_transform([data["Extracurricular Activities"]])[0] 
    df=pd.DataFrame([data])
    df_tranformed=scaler.transform(df)
    return df_tranformed

def predict_data(data):
    model,scaler,le=load_model()
    processed_data=preprocessing_input_data(data,scaler,le)
    prediction=model.predict(processed_data)
    return prediction



def main():
    st.title("Student performance prediction")
    st.write("Enter your data to get a prediction for your performance")

    hours_studied=st.number_input("Hours Studied",min_value=1,max_value=10,value=5)
    previous_score=st.number_input("Previous Scores",min_value=40,max_value=100,value=50)
    extra=st.selectbox("Extra_curricular Activities",['Yes','No'])
    sleeping_hour=st.number_input("Sleeping Hours",min_value=4,max_value=10,value=6)
    number_of_papers_solved=st.number_input("Number of Question Papers Solved",min_value=0,max_value=100,value=5)

    if st.button("predict_your_score"):
        user_data={
            "Hours Studied":hours_studied,
            "Previous Scores":previous_score,
            "Extracurricular Activities":extra,
            "Sleep Hours":sleeping_hour,
            "Sample Question Papers Practiced":number_of_papers_solved

        }
        prediction=predict_data(user_data)
        st.success(f"your prediction result is {prediction}")

 


if __name__ == "__main__":
    main()

