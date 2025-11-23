import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Credit Score Prediction",
    layout="wide"
)

# Title
st.title("Credit Score Prediction")
st.markdown("Enter your financial information below to get credit score predictions.")

@st.cache_resource
def load_model_and_preprocessing():
    try:
        base_path = Path(__file__).parent
        
        # Load LightGBM model
        model_path = base_path / 'best_lgbm.pkl'
        
        if not model_path.exists():
            st.error("LightGBM model file not found. Please ensure 'best_lgbm.pkl' exists.")
            return None, None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Extract feature names from LightGBM model
        if hasattr(model, 'feature_name_'):
            feature_names = list(model.feature_name_)
        else:
            feature_names = None
        
        return model, feature_names
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, feature_names = load_model_and_preprocessing()

if model is None:
    st.stop()

def make_prediction(form_data, model, feature_names):
    try:
        if feature_names is None:
            return None, None, "Feature names not available"
        
        # Create a DataFrame with all feature columns
        input_df = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Fill in the numerical features
        for feature, value in form_data.items():
            if feature in input_df.columns:
                input_df[feature] = value
            else:
                # Try to find similar feature names (case-insensitive, handle variations)
                matching_cols = [col for col in input_df.columns if col.lower() == feature.lower() or 
                                col.lower().replace('_', '') == feature.lower().replace('_', '')]
                if matching_cols:
                    input_df[matching_cols[0]] = value
        
        if 'Credit_Mix' in form_data:
            credit_mix_map = {'Bad': 0, 'Standard': 1, 'Good': 2}
            if 'Credit_Mix' in input_df.columns:
                input_df['Credit_Mix'] = credit_mix_map.get(form_data['Credit_Mix'], 1)
        
        if 'Payment_of_Min_Amount' in form_data:
            payment_col = f"Payment_of_Min_Amount_{form_data['Payment_of_Min_Amount']}"
            if payment_col in input_df.columns:
                input_df[payment_col] = 1
        
        if 'Payment_Behaviour' in form_data:
            behaviour_col = f"Payment_Behaviour_{form_data['Payment_Behaviour']}"
            if behaviour_col in input_df.columns:
                input_df[behaviour_col] = 1
        
        if 'Occupation' in form_data:
            occupation_col = f"Occupation_{form_data['Occupation']}"
            if occupation_col in input_df.columns:
                input_df[occupation_col] = 1
        
        if 'Annual_Income' in form_data and 'Outstanding_Debt' in form_data:
            if 'debt_ratio' in input_df.columns:
                input_df['debt_ratio'] = form_data['Outstanding_Debt'] / (form_data['Annual_Income'] + 1)
        
        if 'Total_EMI_per_month' in form_data and 'Monthly_Inhand_Salary' in form_data:
            if 'monthly_liabilities_ratio' in input_df.columns:
                input_df['monthly_liabilities_ratio'] = form_data['Total_EMI_per_month'] / (form_data['Monthly_Inhand_Salary'] + 1)
        
        if 'Num_of_Loan' in form_data and 'Annual_Income' in form_data:
            if 'loan_to_income_ratio' in input_df.columns:
                input_df['loan_to_income_ratio'] = form_data['Num_of_Loan'] / (form_data['Annual_Income'] + 1)
        
        if 'Monthly_Inhand_Salary' in form_data and 'Total_EMI_per_month' in form_data:
            if 'salary_to_EMI_ratio' in input_df.columns:
                input_df['salary_to_EMI_ratio'] = form_data['Monthly_Inhand_Salary'] / (form_data['Total_EMI_per_month'] + 1)
        
        if 'Monthly_Balance' in form_data and 'Monthly_Inhand_Salary' in form_data:
            if 'monthly_saving_ratio' in input_df.columns:
                input_df['monthly_saving_ratio'] = form_data['Monthly_Balance'] / (form_data['Monthly_Inhand_Salary'] + 1)
        
        if 'Outstanding_Debt' in form_data and 'Changed_Credit_Limit' in form_data:
            if 'debt_to_credit_ratio' in input_df.columns:
                input_df['debt_to_credit_ratio'] = form_data['Outstanding_Debt'] / (form_data.get('Changed_Credit_Limit', 10000) + 1)
        
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        credit_score_map = {0: 'Poor', 1: 'Standard', 2: 'Good'}
        predicted_label = credit_score_map.get(prediction, 'Unknown')
        
        return predicted_label, probabilities, None
    
    except Exception as e:
        return None, None, str(e)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Enter Your Information")
    form_data = {}
    numerical_features = {
        'Annual_Income': {'label': 'Annual Income ($)', 'min': 0, 'max': 500000, 'value': 50000, 'step': 1000},
        'Monthly_Inhand_Salary': {'label': 'Monthly In-hand Salary ($)', 'min': 0, 'max': 50000, 'value': 4000, 'step': 100},
        'Outstanding_Debt': {'label': 'Outstanding Debt ($)', 'min': 0, 'max': 100000, 'value': 5000, 'step': 100},
        'Credit_Utilization_Ratio': {'label': 'Credit Utilization Ratio (%)', 'min': 0, 'max': 100, 'value': 30, 'step': 1},
        'Num_of_Delayed_Payment': {'label': 'Number of Delayed Payments', 'min': 0, 'max': 50, 'value': 0, 'step': 1},
        'Num_of_Loan': {'label': 'Number of Loans', 'min': 0, 'max': 20, 'value': 2, 'step': 1},
        'Num_Credit_Card': {'label': 'Number of Credit Cards', 'min': 0, 'max': 20, 'value': 2, 'step': 1},
        'Interest_Rate': {'label': 'Interest Rate (%)', 'min': 0.0, 'max': 30.0, 'value': 10.0, 'step': 0.5},
        'Num_Credit_Inquiries': {'label': 'Number of Credit Inquiries', 'min': 0, 'max': 20, 'value': 2, 'step': 1},
        'Total_EMI_per_month': {'label': 'Total EMI per Month ($)', 'min': 0, 'max': 5000, 'value': 500, 'step': 50},
        'Age': {'label': 'Age', 'min': 18, 'max': 90, 'value': 35, 'step': 1},
        'Credit_History_Age': {'label': 'Credit History Age (months)', 'min': 0, 'max': 600, 'value': 60, 'step': 6},
        'Changed_Credit_Limit': {'label': 'Changed Credit Limit ($)', 'min': 0, 'max': 50000, 'value': 10000, 'step': 500},
        'Delay_from_due_date': {'label': 'Delay from Due Date (days)', 'min': -30, 'max': 90, 'value': 0, 'step': 1},
        'Amount_invested_monthly': {'label': 'Amount Invested Monthly ($)', 'min': 0, 'max': 5000, 'value': 500, 'step': 50},
        'Monthly_Balance': {'label': 'Monthly Balance ($)', 'min': 0, 'max': 10000, 'value': 1000, 'step': 100},
        'Num_Bank_Accounts': {'label': 'Number of Bank Accounts', 'min': 0, 'max': 10, 'value': 2, 'step': 1},
    }
    
    categorical_features = {
        'Credit_Mix': {
            'label': 'Credit Mix',
            'options': ['Bad', 'Standard', 'Good'],
            'value': 'Standard'
        },
        'Payment_of_Min_Amount': {
            'label': 'Payment of Minimum Amount',
            'options': ['No', 'Yes', 'NM'],
            'value': 'Yes'
        },
        'Payment_Behaviour': {
            'label': 'Payment Behaviour',
            'options': ['Low_spent_Small_value_payments', 'High_spent_Medium_value_payments',
                       'Low_spent_Medium_value_payments', 'High_spent_Large_value_payments',
                       'High_spent_Small_value_payments', 'Low_spent_Large_value_payments', 'Unknown'],
            'value': 'Low_spent_Small_value_payments'
        },
        'Occupation': {
            'label': 'Occupation',
            'options': ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Doctor', 'Lawyer', 
                       'Manager', 'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect',
                       'Developer', 'Journalist', 'Designer', 'Other', 'Unknown'],
            'value': 'Engineer'
        }
    }
    
    for feature, config in numerical_features.items():
        form_data[feature] = st.number_input(
            config['label'],
            min_value=config['min'],
            max_value=config['max'],
            value=config['value'],
            step=config['step'],
            key=feature
        )
    
    for feature, config in categorical_features.items():
        form_data[feature] = st.selectbox(
            config['label'],
            options=config['options'],
            index=config['options'].index(config['value']),
            key=feature
        )

with col2:
    st.subheader("Real-Time Prediction")
    predicted_label, probabilities, error = make_prediction(form_data, model, feature_names)
    
    if error:
        st.error(f"Error making prediction: {error}")
    elif predicted_label:
        st.markdown(f"### Predicted Credit Score: **{predicted_label}**")
        st.markdown("#### Prediction Confidence:")
        prob_df = pd.DataFrame({
            'Category': ['Poor', 'Standard', 'Good'],
            'Probability': probabilities
        })
        prob_df = prob_df.set_index('Category')
        prob_df.index = pd.CategoricalIndex(prob_df.index, categories=['Poor', 'Standard', 'Good'], ordered=True)
        st.bar_chart(prob_df)
        
        prob_cols = st.columns(3)
        score_labels = ['Poor', 'Standard', 'Good']
        
        for i, (col, label, prob) in enumerate(zip(prob_cols, score_labels, probabilities)):
            with col:
                st.metric(label, f"{prob*100:.1f}%")
    else:
        st.info("Start entering your information on the left to see real-time predictions!")
