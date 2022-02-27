#import package
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.naive_bayes import GaussianNB

st.title("Welcome to the HR Employee Attrition App")

#checking the data
st.write("This is an application for predict is your employee have chance for resign or not based on your input.")
st.write("Fill out the required data below and see the chances of resign.")

bus_tra = st.selectbox("How likely the employee do some working travel?",("Non-Travel","TravelRarely","TravelFrequently"))
mar_sta = st.selectbox("What is the marital status of the employee?",("Married","Divorce","Single"))
ovr_time = st.selectbox("Is this employee working overtime?",("No","Yes"))
job_rol = st.selectbox("What is the job role of the employee?",("Research Director",
                                                                "Manager",
                                                                "Healthcare Representative",
                                                                "Manufacturing Director",
                                                                "Human Resources",
                                                                "Sales Representative",
                                                                "Research Scientist",
                                                                "Sales Executive",
                                                                "Laboratory Technician"))
work_years = st.slider("How long work experience the employee have? (in years)",0,40,10)
job_level = st.slider("What is the job level of the employee? (1 for staff, increase until 5 is C-level)",1,5,2)
curr_years = st.slider("How long work experience in the current company? (in years)",0,18,3)
income = st.slider("How much the monthly income of the employee",1000,19999,4919)
age = st.slider("Age of employee",18,60,36)

bus_tra = np.where(bus_tra=='Non-Travel', 0,
            np.where(bus_tra=='TravelRarely', 1, 2))
mar_sta = np.where(mar_sta=='Married', 0,
                         np.where(mar_sta=='Divorce', 1, 2))
ovr_time = np.where(ovr_time=="No",0,1)
job_rol = np.where(job_rol == 'Research Director', 0,
                  np.where(job_rol == 'Manager', 1,
                  np.where(job_rol == 'Healthcare Representative', 2,
                  np.where(job_rol == 'Manufacturing Director', 3,
                  np.where(job_rol == 'Human Resources', 4,
                  np.where(job_rol == 'Sales Representative', 5,
                  np.where(job_rol == 'Research Scientist', 6,
                  np.where(job_rol == 'Sales Executive', 7, 8))))))))

#import model
filename = 'naive_bayes_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
predictions = loaded_model.predict_proba([[bus_tra, mar_sta, ovr_time, job_rol, work_years, job_level, curr_years, income, age]])[0][1]

if st.button("Predict"):
    st.header("This employee have {:.2f} % chances of attrition.".format((predictions*100)))
