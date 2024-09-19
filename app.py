import pandas as pd
import streamlit as st
import pickle as pk

# Load pre-trained Loan Prediction model (replace 'loan_model.pkl' with your actual file path)
model = pk.load(open('clf.pkl', 'rb'))

# Streamlit header
st.header('Loan Approval Prediction')

# Define mappings for categorical values
education_mapping = {'Graduate': 1, 'Not Graduate': 0}
self_employed_mapping = {'Yes': 1, 'No': 0}

# Define outcome mappings
outcome_mapping = {0: 'Loan Approved', 1: 'Loan Not Approved'}

# UI components for input
no_of_dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, value=0)
education = st.selectbox('Education', list(education_mapping.keys()))
self_employed = st.selectbox('Self Employed', list(self_employed_mapping.keys()))
income_annum = st.number_input('Annual Income (in INR)', min_value=0, value=0)
loan_amount = st.number_input('Loan Amount (in INR)', min_value=0, value=0)
loan_term = st.number_input('Loan Term (in months)', min_value=0, value=0)
cibil_score = st.number_input('CIBIL Score', min_value=0, value=0)

# Convert categorical values to numerical values using mappings
input_data = pd.DataFrame(
    [[no_of_dependents, education_mapping[education], self_employed_mapping[self_employed], 
      income_annum, loan_amount, loan_term, cibil_score]],
    columns=["no_of_dependents", "education", "self_employed", "income_annum", 
             "loan_amount", "loan_term", "cibil_score"]
)

# Display input data for debugging
st.write("Input Data:")
st.write(input_data)

# Convert DataFrame to NumPy array for prediction
input_data_array = input_data.values

# Prediction button
if st.button('Predict'):
    try:
        # Predict loan approval status
        prediction = model.predict(input_data_array)
        st.write("Raw Prediction Output: ")
        st.write(prediction)

        result = outcome_mapping.get(prediction[0], 'Unknown Outcome')

        # Output prediction
        st.markdown(f'The loan application is: {result}')
        
        # Optional: Show detailed prediction info
        if prediction[0] == 1:
            st.info("The model predicts 'Loan Rejected' . Consider reviewing the input features or retraining the model.")
        else:
            st.success("Congraluations your 'Loan Approved'.")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
