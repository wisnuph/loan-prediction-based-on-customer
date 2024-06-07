import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

scaler = joblib.load('scaler_loan.pkl')
model = joblib.load('random_forest_model.pkl')

def predict(input_data):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    return prediction, probability

def batch_predict(df):
    df_predict = df.copy()
    df_predict = df_predict.drop(columns=['CITY', 'ID', 'Profession', 'STATE'], errors='ignore')
    df_predict['Married/Single'] = df_predict['Married/Single'].replace({"single": 0, "married": 1})
    df_predict['House_Ownership'] = df_predict['House_Ownership'].replace({"norent_noown": 0, "rented": 1, 'owned':2})
    df_predict['Car_Ownership'] = df_predict['Car_Ownership'].replace({"no": 0, "yes": 1})
    df_predict['prod_yrs_left'] = np.maximum(64 - df_predict['Age'], 0)
    expected_columns = ['Income', 'Age', 'Experience', 'Married/Single', 'House_Ownership', 'Car_Ownership', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS', 'prod_yrs_left']
    df_predict = df_predict[expected_columns]
    input_scaled = scaler.transform(df_predict)
    predictions = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    df_predict['Prediction'] = predictions 
    df_predict['Persentase'] = [f'{prob[0]*100:.2f} %' if pred == 0 else f'{prob[1]*100:.2f} %' for pred, prob in zip(predictions, probabilities)]
    df_predict['Prediction'] = df_predict['Prediction'].replace({0: "Eligible", 1: "Not Eligible"})
    df_predict['Married/Single'] = df_predict['Married/Single'].replace({0: "single", 1: "married"})
    df_predict['House_Ownership'] = df_predict['House_Ownership'].replace({0: "norent_noown", 1: "rented", 2: 'owned'})
    df_predict['Car_Ownership'] = df_predict['Car_Ownership'].replace({0: "no", 1: "yes"})
    df_merge = pd.concat([df['ID'], df_predict[['Income', 'Age', 'Experience', 'Married/Single', 'House_Ownership', 'Car_Ownership']], df[['Profession', 'CITY', 'STATE']], df_predict[['CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS', 'Prediction', 'Persentase']]], axis=1)
    return predictions, probabilities, df_merge

navbar_container = st.container()

with navbar_container:
    col1, col2 = st.columns([1, 10])
    with col1:
        logo_img = st.image('logo_psm.jpg', width=100)
    with col2:
        st.markdown("<div style='margin-top: 35px;'></div>", unsafe_allow_html=True)
        options = ["Home", "Tools", "About Us", "Contact"]
        selected = option_menu(
            menu_title=None,
            options=options,
            icons=["house", "tools", "info-circle", "envelope"],
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa", "display": "flex", "justify-content": "flex-start"},
                "icon": {"color": "orange", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee", "color": "black", "font-weight": "bold"},
                "nav-link-selected": {"background-color": "#007bff"},
                "nav-link:hover": {"background-color": "#f7f7f7"},
                ".nav-link::before": {"content": "", "display": "block", "width": "100%", "height": "2px", "background-color": "#ccc", "position": "absolute", "top": "0", "left": "0"},
                ".nav-link-selected::before": {"background-color": "#007bff"}
            }
                
        )

content_container = st.container()

with content_container:
    if selected == "Home":
        st.markdown("")
        st.title("Early Warning Loan System")
        st.markdown("")
        st.markdown(f"<p style='font-size:20px;'><b>The Early Warning Loan System</b> <i>is a tool designed to assist individuals in understanding their likelihood of obtaining a loan. This system is trained on a comprehensive dataset of financial and demographic data, and can provide estimates of loan eligibility probability.</i></p>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("")
        st.markdown(f"<p style='font-size:18px; font-weight:bold;'>Statistics:</p>", unsafe_allow_html=True)
        st.write("* Over 10,000 users have used our system to date")
        st.write("* Our system has a accuracy rate of 95% in predicting loan eligibility")
        st.write("* We have partnered with > 20 major banks to provide loan options to our users")

        st.write("**What our users say:**")
        st.write('"The Early Warning Loan System helped me understand my loan options and make an informed decision. I was able to get a loan with a lower interest rate than I expected!" - Dedi Samosir.')

        st.markdown("")
        st.markdown(f"<p style='font-size:20px;'>Ready to get started? Click on the <b>Tools</b> menu to analyze your loan eligibility today!</p>", unsafe_allow_html=True)
        st.markdown("")
        st.markdown(f"<p style='font-size:20px;'>Or <b>watch</b> a video tutorial below:</p>", unsafe_allow_html=True)

    elif selected == "Tools":
            st.markdown("")
            st.title("Tools")

            tool_option = st.selectbox("Choose an option", ["Analyze Only One Customer", "Analyze Many Customers"])

            if tool_option == "Analyze Only One Customer":
                st.header("Analyze Only One Customer")

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

            elif tool_option == "Analyze Many Customers":
                st.header("Analyze Many Customers")

                uploaded_file = st.file_uploader("Choose a CSV file (Must be header in [ID, Income, Age, Experience, Married/Single, House_Ownership, Car_Ownership, Profession, CITY, STATE, CURRENT_JOB_YRS, CURRENT_HOUSE_YRS])", type="csv")

                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    st.write(df)

                    if st.button("**Predict for All**"):
                        predictions, probabilities, df = batch_predict(df)
                        st.write("**Result**")
                        st.write(df)

                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Data",
                            data=csv,
                            file_name='predictions.csv',
                            mime='text/csv',
                        )

    elif selected == "About Us":
            st.markdown("")
            st.title("About Us")
            st.write("**OUR TEAM:**")
            st.write("Our team consists of experts in data science and software development. We are committed to creating a user-friendly and reliable tool that can empower individuals to make informed financial decisions.")
            st.image('our_team.jpg', width=1200)

    elif selected == "Contact":
            st.markdown("")
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
