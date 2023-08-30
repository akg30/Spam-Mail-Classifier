# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:53:34 2023

@author: Hp
"""

import streamlit as st
import numpy as np
import pickle as pk

loaded_model=pk.load(open('trained_model.sav','rb'))
tfid_model=pk.load(open('tfid_model.sav','rb'))

def mail(input_mail):
    input_mail_new=[input_mail]
    input_mail_features=tfid_model.transform(input_mail_new)
    predict=loaded_model.predict(input_mail_features)
    if predict==1:
        return 'mail is spam'
    else:
        return 'mail is not spam'

def main():
    st.title('Spam Mail Prediction Using Machine Learning')
    mail_user=st.text_area('Enter Your Mail')
    mail_check=' '
    if st.button('Click Here To Check Whether Your Mail Is Spam Or Not'):
        mail_check=mail(mail_user)
    st.success(mail_check)
    st.markdown('##### Exploratory Data Analysis Done And Machine Learning Deployed By "Anubhav Kumar Gupta"')

if __name__=="__main__":
    main()
              
