import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('rf_model.pkl','rb'))


def predict_default(LIMIT_BAL, EDUCATION, MARRIAGE, AGE, PAY_1, BILL_AMT1, BILL_AMT2, BILL_AMT3,
 BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6):
    input=np.array([[LIMIT_BAL, EDUCATION, MARRIAGE, AGE, PAY_1, BILL_AMT1, BILL_AMT2, BILL_AMT3,
 BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]]).astype(np.float64)
    prediction=model.predict(input)
    return int(prediction)

def main():
    st.title("Default payment")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Bank Account Default Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
 
    LIMIT_BAL = st.text_input("LIMIT_BAL")
    EDUCATION = st.text_input("EDUCATION")
    MARRIAGE = st.text_input("MARRIAGE")
    AGE = st.text_input("AGE")
    PAY_1 = st.text_input("PAY1")
    BILL_AMT1 = st.text_input("BILL_AMT1")
    BILL_AMT2 = st.text_input("BILL_AMT2")
    BILL_AMT3 = st.text_input("BILL_AMT3")
    BILL_AMT4 = st.text_input("BILL_AMT4")
    BILL_AMT5 = st.text_input("BILL_AMT5")
    BILL_AMT6 = st.text_input("BILL_AMT6")
    PAY_AMT1 = st.text_input("PAY_AMT1")
    PAY_AMT2 = st.text_input("PAY_AMT2")
    PAY_AMT3 = st.text_input("PAY_AMT3")
    PAY_AMT4 = st.text_input("PAY_AMT4")
    PAY_AMT5 = st.text_input("PAY_AMT5")
    PAY_AMT6 = st.text_input("PAY_AMT6")
    
    

    if st.button("Predict"):
        output=predict_default(LIMIT_BAL, EDUCATION, MARRIAGE, AGE, PAY_1, BILL_AMT1, BILL_AMT2, BILL_AMT3,
 BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6)

        if output == 1:
            st.success('Default')
    
        else:
            st.success('Not Default')
            

if __name__=='__main__':
    main()