import streamlit as st
import pandas as pd
import pickle # Use pickle instead of joblib
import os
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import holidays
import networkx as nx
import re
# import streamlit_ngrok # Removed as it caused installation issues

# Download NLTK data if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Load the saved model and preprocessors
model_path = '/content/fraud_detection_model/best_model.pkl'
scaler_path = '/content/fraud_detection_model/scaler.pkl'
tfidf_path = '/content/fraud_detection_model/tfidf_vectorizer.pkl'
features_path = '/content/fraud_detection_model/feature_columns.pkl'

try:
    with open(model_path, 'rb') as f: # Use pickle for loading
        best_model = pickle.load(f)
    with open(scaler_path, 'rb') as f: # Use pickle for loading
        scaler = pickle.load(f)
    with open(tfidf_path, 'rb') as f: # Use pickle for loading
        tfidf = pickle.load(f)
    with open(features_path, 'rb') as f: # Use pickle for loading
        features_for_splitting = pickle.load(f)
    st.success("Model and preprocessors loaded successfully!")
except FileNotFoundError:
    st.error("Error loading model or preprocessors. Make sure the files exist.")
    st.stop() # Stop the app if essential files is missing

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define preprocessing function (should match the steps in your notebook)
def preprocess_data(df, scaler, tfidf, features_for_splitting):
    # Ensure date columns are in datetime format
    date_cols = ['Incident_Date', 'Claim_Submission_Date', 'Policy_Start_Date', 'Policy_End_Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    # Re-engineer features based on the notebook's feature engineering steps
    # Handle potential missing values in 'Adjuster_Notes' before TF-IDF
    if 'Adjuster_Notes' in df.columns:
        df['Adjuster_Notes'] = df['Adjuster_Notes'].fillna('')

        # Create TF-IDF features
        # Need to handle cases where new data might have words not in the original TF-IDF vocabulary
        tfidf_matrix = tfidf.transform(df['Adjuster_Notes'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{word}" for word in tfidf.get_feature_names_out()])
        df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    else:
         # If 'Adjuster_Notes' is missing, create dummy TF-IDF columns with zeros
         tfidf_df = pd.DataFrame(0.0, index=df.index, columns=[f"tfidf_{word}" for word in tfidf.get_feature_names_out()])
         df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)



    # Recreate graph-based features (requires re-building the graph for the new data)
    # This is a simplified approach; in a real application, you might update the graph
    # with new data or use a different approach for graph features in prediction.
    # For now, we'll add dummy centrality columns if Location or Customer_Name are present
    if 'Customer_Name' in df.columns and 'Location' in df.columns:
        try:
            G = nx.Graph()
            G.add_nodes_from(df['Customer_Name'].dropna().unique(), bipartite='customer')
            G.add_nodes_from(df['Location'].dropna().unique(), bipartite='location')
            edges = list(zip(df['Customer_Name'].dropna(), df['Location'].dropna()))
            G.add_edges_from(edges)
            centrality = nx.degree_centrality(G)
            centrality_df = pd.DataFrame.from_dict(centrality, orient='index', columns=['Centrality_Score'])
            centrality_df['Name'] = centrality_df.index
            customer_centrality_map = centrality_df[centrality_df['Name'].isin(df['Customer_Name'].dropna().unique())].set_index('Name')['Centrality_Score'].to_dict()
            location_centrality_map = centrality_df[centrality_df['Name'].isin(df['Location'].dropna().unique())].set_index('Name')['Centrality_Score'].to_dict()

            df['Customer_Centrality'] = df['Customer_Name'].map(customer_centrality_map).fillna(0) # Fill NaN for new customers
            df['Location_Centrality'] = df['Location'].map(location_centrality_map).fillna(0) # Fill NaN for new locations
        except Exception as e:
            st.warning(f"Could not re-engineer graph-based features: {e}. Adding dummy columns.")
            df['Customer_Centrality'] = 0.0
            df['Location_Centrality'] = 0.0
    else:
         st.warning("Customer_Name or Location column missing. Cannot re-engineer graph-based features. Adding dummy columns.")
         df['Customer_Centrality'] = 0.0
         df['Location_Centrality'] = 0.0


    # Recreate sentiment features
    if 'Adjuster_Notes' in df.columns:
        df['Sentiment_Score'] = df['Adjuster_Notes'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
        df['Negative_Tone_Flag'] = (df['Sentiment_Score'] < -0.5).astype(int) # Ensure it's int like in training
    else:
        st.warning("'Adjuster_Notes' column missing. Cannot re-engineer sentiment features. Adding dummy columns.")
        df['Sentiment_Score'] = 0.0
        df['Negative_Tone_Flag'] = 0


    # Recreate date-based features
    if 'Incident_Date' in df.columns:
        # Need the range of years from the original training data for holidays
        # This is a simplification; ideally, you'd handle holidays dynamically or precompute a larger range
        start_year = 2023 # Assuming from your notebook's last run
        end_year = 2025   # Assuming from your notebook's last run
        years = range(start_year, end_year + 1)
        nigerian_holidays = holidays.Nigeria(years=years)

        df['Incident_on_Holiday'] = df['Incident_Date'].apply(lambda date: int(date in nigerian_holidays) if pd.notnull(date) else 0) # Handle NaT
        df['Incident_on_Weekend'] = df['Incident_Date'].dt.dayofweek.apply(lambda x: 1 if pd.notnull(x) and x >= 5 else 0) # Handle NaT
    else:
        st.warning("'Incident_Date' column missing. Cannot re-engineer date-based features. Adding dummy columns.")
        df['Incident_on_Holiday'] = 0
        df['Incident_on_Weekend'] = 0

    if 'Claim_Submission_Date' in df.columns:
         df['Claim_Submission_on_Weekend'] = df['Claim_Submission_Date'].dt.dayofweek.apply(lambda x: 1 if pd.notnull(x) and x >= 5 else 0) # Handle NaT
    else:
         st.warning("'Claim_Submission_Date' column missing. Cannot re-engineer claim submission weekend feature. Adding dummy column.")
         df['Claim_Submission_on_Weekend'] = 0


    if 'Claim_Submission_Date' in df.columns and 'Incident_Date' in df.columns:
        df['Days_to_Claim_Submission'] = (df['Claim_Submission_Date'] - df['Incident_Date']).dt.days.fillna(-1) # Fill NaNs with -1 or another indicator
        df['Late_Claim_Submission'] = (df['Days_to_Claim_Submission'] >= 90).astype(int)
    else:
        st.warning("'Claim_Submission_Date' or 'Incident_Date' missing. Cannot re-engineer claim submission delay features. Adding dummy columns.")
        df['Days_to_Claim_Submission'] = -1
        df['Late_Claim_Submission'] = 0


    if 'Policy_End_Date' in df.columns and 'Policy_Start_Date' in df.columns:
        df['Policy_Duration_Days'] = (df['Policy_End_Date'] - df['Policy_Start_Date']).dt.days.fillna(-1) # Fill NaNs with -1
    else:
         st.warning("'Policy_End_Date' or 'Policy_Start_Date' missing. Cannot re-engineer policy duration. Adding dummy column.")
         df['Policy_Duration_Days'] = -1


    # Recreate Claim Count 2 Years and Frequent Claimant
    if 'Policy_Number' in df.columns and 'Claim_Submission_Date' in df.columns and 'Policy_Start_Date' in df.columns:
        claims_within_2years = df[
            df.apply(
                lambda row: pd.notnull(row['Claim_Submission_Date']) and pd.notnull(row['Policy_Start_Date']) and (row['Claim_Submission_Date'] - row['Policy_Start_Date']).days <= 730, axis=1
            )
        ].copy()
        claim_counts_2years = claims_within_2years.groupby('Policy_Number').size().reset_index(name='Claim_Count_2Years')
        df = df.merge(claim_counts_2years, on='Policy_Number', how='left')
        df['Claim_Count_2Years'] = df['Claim_Count_2Years'].fillna(0)
        df['Frequent_Claimant'] = (df['Claim_Count_2Years'] > 3).astype(int)
    else:
         st.warning("Required columns for Claim Count 2 Years/Frequent Claimant missing. Adding dummy columns.")
         df['Claim_Count_2Years'] = 0
         df['Frequent_Claimant'] = 0


    # Recreate High Claim Amount Flag
    if 'Claim_Amount' in df.columns:
        # Need the 90th percentile from the original training data
        # This is a simplification; ideally, you'd save and load the percentile value
        percentile_90 = 454548.349 # Assuming from your notebook's last run
        df['High_Claim_Amount_Flag'] = (df['Claim_Amount'] > percentile_90).astype(int)
    else:
         st.warning("'Claim_Amount' column missing. Cannot re-engineer High Claim Amount Flag. Adding dummy column.")
         df['High_Claim_Amount_Flag'] = 0


    # Recreate Claim vs Premium Ratio
    if 'Claim_Amount' in df.columns and 'Premium_Amount' in df.columns:
         # Handle division by zero or missing premium amounts
         df['Claim_vs_Premium_Ratio'] = df.apply(lambda row: row['Claim_Amount'] / row['Premium_Amount'] if row['Premium_Amount'] > 0 and pd.notnull(row['Premium_Amount']) else 0, axis=1)
    else:
         st.warning("'Claim_Amount' or 'Premium_Amount' missing. Cannot re-engineer Claim vs Premium Ratio. Adding dummy column.")
         df['Claim_vs_Premium_Ratio'] = 0.0

    # Recreate Customer Claim Frequency
    if 'Customer_Name' in df.columns:
        customer_claim_counts = df.groupby('Customer_Name').size().reset_index(name='Customer_Claim_Count')
        df = df.merge(customer_claim_counts, on='Customer_Name', how='left')
        df['Frequent_Customer_Claimant'] = (df['Customer_Claim_Count'] > 2).astype(int)
    else:
         st.warning("'Customer_Name' column missing. Cannot re-engineer Customer Claim Frequency. Adding dummy column.")
         df['Customer_Claim_Count'] = 0
         df['Frequent_Customer_Claimant'] = 0

    # Recreate Prior Fraudulent Claim
    # This feature is based on whether a customer had a fraudulent claim *in the training data*.
    # For new data, you would need a database of known fraudulent customers.
    # As a simplification here, we'll assume no prior fraudulent claims for new customers unless they were in the training data.
    # A more robust approach would involve querying a database.
    # For this app, we'll add a dummy column or try to map from existing data if Customer_Name exists.
    if 'Customer_Name' in df.columns and 'Prior_Fraudulent_Claim' in features_for_splitting: # Check if it was a feature in training
         # If Prior_Fraudulent_Claim was engineered in the notebook, it might exist.
         # If not, or for new customers, this needs a lookup mechanism.
         # Assuming for simplicity that if the column exists, we use it. If not, all new claims are treated as no prior fraud.
         if 'Prior_Fraudulent_Claim' not in df.columns:
              df['Prior_Fraudulent_Claim'] = 0 # Assume no prior fraud for new data
    else:
        st.warning("'Customer_Name' column missing or 'Prior_Fraudulent_Claim' not in original data/training features. Cannot re-engineer Prior Fraudulent Claim. Adding dummy column.")
        df['Prior_Fraudulent_Claim'] = 0


    # Recreate claims within 2 months of policy start/end dates
    if 'Claim_Submission_Date' in df.columns and 'Policy_Start_Date' in df.columns:
         df['Claim_Within_2Months_of_Start'] = ((df['Claim_Submission_Date'] - df['Policy_Start_Date']).dt.days <= 60
         ).astype(int)
    else:
         st.warning("Required columns for Claim Within 2 Months of Start missing. Adding dummy column.")
         df['Claim_Within_2Months_of_Start'] = 0


    if 'Claim_Submission_Date' in df.columns and 'Policy_End_Date' in df.columns:
         df['Claim_Within_2Months_of_End'] = ( ((df['Policy_End_Date'] - df['Claim_Submission_Date']).dt.days <= 60) &
             ((df['Policy_End_Date'] - df['Claim_Submission_Date']).dt.days >= 0)
         ).astype(int)
    else:
        st.warning("Required columns for Claim Within 2 Months of End missing. Adding dummy column.")
        df['Claim_Within_2Months_of_End'] = 0


    # Handle Categorical Features - Need to apply one-hot encoding consistently
    # Identify categorical columns from the original training data features that are still objects
    # This requires knowing the original categorical columns before encoding
    # As a simplification, we'll re-identify object columns here and encode,
    # but this might create issues if new data has categories not seen in training.
    # A robust solution would involve saving and loading the OneHotEncoder fitted on training data.

    # For now, identify object columns that are *not* dates or text notes
    categorical_cols_to_encode = [col for col in df.columns if df[col].dtype == 'object' and col not in date_cols + ['Customer_Name', 'Customer_Email', 'Customer_Phone', 'Adjuster_Notes', 'Policy_Number', 'Claim_ID']]

    if categorical_cols_to_encode:
        # Apply one-hot encoding
        df = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)


    # Align columns with the features the model was trained on
    # Add missing columns (features present in training but not in new data) with a value of 0
    for col in features_for_splitting:
        if col not in df.columns:
            df[col] = 0

    # Ensure the order of columns is the same as during training
    df = df[features_for_splitting]


    # Scale numerical features - Apply the *fitted* scaler
    # Need to identify which of the 'features_for_splitting' are numerical and were scaled during training
    # This requires a more robust way to track which features were originally numerical and scaled.
    # Let's identify numerical features from the final 'features_for_splitting' list
    # that are not dummy variables or explicitly non-numerical engineered features.
    numerical_features_to_scale = [col for col in features_for_splitting if df[col].dtype in ['int64', 'float64'] and not col.startswith('tfidf_') and col not in ['Customer_Centrality', 'Location_Centrality', 'Sentiment_Score', 'Negative_Tone_Flag', 'Claim_Count_2Years', 'Frequent_Claimant', 'High_Claim_Amount_Flag', 'Claim_vs_Premium_Ratio', 'Customer_Claim_Count', 'Frequent_Customer_Claimant', 'Prior_Fraudulent_Claim', 'Claim_Within_2Months_of_Start', 'Claim_Within_2Months_of_End']]

    if numerical_features_to_scale:
         try:
              # Ensure the input to scaler.transform is always a 2D array
              df[numerical_features_to_scale] = scaler.transform(df[numerical_features_to_scale].values) # Use .values to get NumPy array
         except Exception as e:
              st.warning(f"Error applying scaler: {e}. Skipping numerical scaling.")
              # Handle error - maybe skip scaling or log it
    else:
         st.info("No numerical features identified for scaling.")


    return df

# Streamlit App
st.title("Fraud Detection Application")

st.write("""
Upload your claims data (CSV format) to get fraud predictions.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Read the uploaded file
        input_df = pd.read_csv(uploaded_file)
        st.write("Original Data:")
        st.dataframe(input_df.head())

        # Preprocess the data
        st.write("Preprocessing data...")
        processed_df = preprocess_data(input_df.copy(), scaler, tfidf, features_for_splitting) # Pass a copy to avoid modifying original

        st.write("Preprocessing complete. Features for prediction:")
        st.dataframe(processed_df.head())
        st.write(f"Shape of processed data: {processed_df.shape}")


        # Make predictions
        st.write("Making predictions...")
        predictions = best_model.predict(processed_df)
        predictions_proba = best_model.predict_proba(processed_df)[:, 1] # Probability of fraud

        # Add predictions to the original dataframe for display
        input_df['Predicted_Fraud_Flag'] = predictions
        input_df['Predicted_Fraud_Probability'] = predictions_proba

        st.write("Predictions complete.")

        # Display results
        st.subheader("Prediction Results")
        st.dataframe(input_df[['Claim_ID', 'Policy_Number', 'Customer_Name', 'Claim_Amount', 'Predicted_Fraud_Flag', 'Predicted_Fraud_Probability']])

        # Summary of predictions
        fraud_count = input_df['Predicted_Fraud_Flag'].sum()
        total_claims = input_df.shape[0]
        st.write(f"Total Claims Processed: {total_claims}")
        st.write(f"Predicted Fraudulent Claims: {fraud_count}")
        st.write(f"Predicted Non-Fraudulent Claims: {total_claims - fraud_count}")

        # Option to download results
        @st.cache_data # Cache the function to avoid re-running unnecessarily
        def convert_df_to_csv(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df_to_csv(input_df)

        st.download_button(
            label="Download Prediction Results (CSV)",
            data=csv,
            file_name='fraud_predictions.csv',
            mime='text/csv',
        )


    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        st.error("Please check the uploaded file format and content.")
