import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pandas as pd
import numpy as np

scaler = joblib.load('scaler_loan.pkl')
model = joblib.load('random_forest_model.pkl')

def predict(input_data):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    return prediction, probability

st.set_page_config(layout="wide")

logo_container = st.markdown("""
<style>
    .logo-container {
        display: flex;
        align-items: center;  }
</style>
<div class="logo-container">
    """, unsafe_allow_html=True)

with logo_container:
    logo_img = st.image('logo_psm.jpg', width=100)

options = ["Home", "About Us", "Contact"]

selected = option_menu(
    menu_title=None,
    options=options,
    icons=["house", "info-circle", "envelope"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa", "display": "flex", "justify-content": "flex-start"},  # Align navbar items to the left
        "icon": {"color": "orange", "font-size": "20px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee", "color": "black", "font-weight": "bold"},
        "nav-link-selected": {"background-color": "#007bff"},
        "nav-link:hover": {"background-color": "#f7f7f7"},
        ".nav-link::before": {"content": "", "display": "block", "width": "100%", "height": "2px", "background-color": "#ccc", "position": "absolute", "top": "0", "left": "0"},
        ".nav-link-selected::before": {"background-color": "#007bff"}
    }
)

if selected == "Home":
    st.title("Early Warning Loan System")
    st.markdown("")
    st.markdown("")

    col1, col2 = st.columns([1, 1])

    with col1:
        name = st.text_input("**Name**")
        profession = st.text_input("**Profession**")
        city = st.text_input("**City**")

        income = st.text_input("**Income**")
        try:
            income = float(income)
            income_valid = True
        except ValueError:
            st.warning("Income input must be a number.")
            income_valid = False

        age = st.text_input("**Age**")
        try:
            age = float(age)
            age_valid = True
        except ValueError:
            st.warning("Age input must be a number.")
            age_valid = False

        experience = st.text_input("**Experience**")
        try:
            experience = float(experience)
            experience_valid = True
        except ValueError:
            st.warning("Experience input must be a number.")
            experience_valid = False

    with col2:
        marital_status = st.selectbox("**Marital Status**", ("Single", "Married"))
        status_code = 1 if marital_status == "Married" else 0

        house_ownership = st.selectbox("**House Ownership**", ("norent_noown", "rented", "owned"))
        house_code = {"norent_noown": 0, "rented": 1, "owned": 2}[house_ownership]

        car_ownership = st.selectbox("**Car Ownership**", ("Yes", "No"))
        car_code = 1 if car_ownership == "Yes" else 0

        current_job_yrs = st.text_input("**Current Job Years**")
        try:
            current_job_yrs = float(current_job_yrs)
            current_job_yrs_valid = True
        except ValueError:
            st.warning("Current Job Years input must be a number.")
            current_job_yrs_valid = False

        current_house_yrs = st.text_input("**Current House Years**")
        try:
            current_house_yrs = float(current_house_yrs)
            current_house_yrs_valid = True
        except ValueError:
            st.warning("Current House Years input must be a number.")
            current_house_yrs_valid = False

        prod_yrs_left = max(64 - age, 0) if age_valid else 0

        if st.button("**Predict status Loan**"):
            if all([income_valid, age_valid, experience_valid, current_job_yrs_valid, current_house_yrs_valid]):
                input_data = {
                    'Income': income,
                    'Age': age,
                    'Experience': experience,
                    'Married/Single': status_code,
                    'House_Ownership': house_code,
                    'Car_Ownership': car_code,
                    'CURRENT_JOB_YRS': current_job_yrs,
                    'CURRENT_HOUSE_YRS': current_house_yrs,
                    'prod_yrs_left': prod_yrs_left  
                }

                prediction, probability = predict(input_data)
                if prediction[0] == 0:
                    st.markdown(f"<p style='color:green; font-size:17px;'>Result: {name} is <b>eligible</b> for loan with probability {probability[0][0]*100:.2f}%</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='color:red; font-size:17px;'>Result: {name} is <b>not eligible</b> for loan with probability {probability[0][1]*100:.2f}%</p>", unsafe_allow_html=True)
            else:
                st.warning("All input must be valid. Check your input again!")

elif selected == "About Us":
    st.title("About Us")
    st.title("**Early Warning Loan System**")
    about_text = """
    The Early Warning Loan System is a tool designed to assist individuals in understanding their likelihood of obtaining a loan. This system is trained on a comprehensive dataset of financial and demographic data, and can provide estimates of loan eligibility probability.

    Our team consists of experts in data science and software development. We are committed to creating a user-friendly and reliable tool that can empower individuals to make informed financial decisions.

    The system is under continuous development, and we are constantly working to improve its accuracy. It's important to note that this system is for informational purposes only and should not be used as a substitute for professional financial advice.
    """

    st.write(about_text)

elif selected == "Contact":
    st.title("Contact")

    st.write("**Email:** [pejuangsabtumalam@gmail.com](mailto:[pejuangsabtumalam@gmail.com])")  

    st.write("**Phone:** (62) 852-1806-0624")  
    st.write("**Address:** Freeyork, North Jakarta") 

    st.write("**Social Media:**")
    st.write("[Link to Facebook Pejuang Sabtu Malam](https://www.facebook.com/wisnuph)") 
    st.write("[Link to Instagram Pejuang Sabtu Malam](https://instagram.com/wisnu_ph)")  

    contact_form = st.form("Contact Form")
    name = contact_form.text_input("Your Name")
    email = contact_form.text_input("Your Email")
    message = contact_form.text_area("Your Message")
    submit_button = contact_form.form_submit_button("Send Message")

    if submit_button:
        st.write("Thank you for contacting us!")
