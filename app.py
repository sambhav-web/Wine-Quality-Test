import streamlit as st
import numpy as np
import pickle as pkl

with open('model.pkl','rb') as f:
  model=pkl.load(f)

with open('scaler.pkl','rb') as f:
  scaler=pkl.load(f)

st.title('Wine Quality Prediction')
st.header('Enter the required Details: ')
fixed_acidity=st.number_input('fixed acidity',0.0,18.0,8)
volatile_acidity=st.number_input('volatile acidity',0.0,3.0,0.52)
citric_acid=st.number_input('citric acid',0.0,1.0,0.42)
residual_sugar=st.number_input('residual sugar',0.0,18.0,2.22)
chlorides=st.number_input('chlorides',0.0,1.0,0.08)
free_sulfur_dioxide=st.number_input('free sulfur dioxide',0.0,75.0,14.0)
total_sulfur_dioxide=st.number_input('total sulfur dioxide',0.0,200.0,38)
density=st.number_input('density',0.0,1.2,0.99)
pH =st.number_input('pH',0.0,4.5,3.31)
sulphates=st.number_input('sulphates',0.0,2.0,0.62)
alcohol=st.number_input('alcohol',0.0,14.0,10.21)
def predict(f_a,v_a,c_a,r_s,c,f_s,t_s,d,pH,s,a):
  input=(f_a,v_a,c_a,r_s,c,f_s,t_s,d,pH,s,a)
  input=np.array(input).reshape(1,-1)
  scaled_input=scaler.transform(input)
  prediction=model.predict(scaled_input)
  return prediction[0]

if st.button('WINE QUALITY'):
  prediction=predict(fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol)
  st.success(f'Quality of Wine Comes out to Q{prediction}')