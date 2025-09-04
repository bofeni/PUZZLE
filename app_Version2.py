import streamlit as st
import pandas as pd
import pickle
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import holidays
import networkx as nx

# Download NLTK data if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Use relative paths (files must be in same folder as app.py)
model_path = 'best_model.pkl'
scaler_path = 'scaler.pkl'
tfidf_path = 'tfidf_vectorizer.pkl'
features_path = 'feature_columns.pkl'

try:
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    with open(features_path, 'rb') as f:
        features_for_splitting = pickle.load(f)
    st.success("Model and preprocessors loaded successfully!")
except FileNotFoundError:
    st.error("Error loading model or preprocessors. Make sure the files exist in the app folder.")
    st.stop()

sia = SentimentIntensityAnalyzer()

def preprocess_data(df, scaler, tfidf, features_for_splitting):
    date_cols = ['Incident_Date', 'Claim_Submission_Date', 'Policy_Start_Date', 'Policy_End_Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    # TF-IDF Adjuster_Notes
    if 'Adjuster_Notes' in df.columns:
        df['Adjuster_Notes'] = df['Adjuster_Notes'].fillna('')
        tfidf_matrix = tfidf.transform(df['Adjuster_Notes'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{word}" for word in tfidf.get_feature_names_out()])
        df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    else:
        tfidf_df = pd.DataFrame(0.0, index=df.index, columns=[f"tfidf_{word}" for word in tfidf.get_feature_names_out()])
        df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    # Graph-based features
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
            df['Customer_Centrality'] = df['Customer_Name'].map(customer_centrality_map).fillna(0)
            df['Location_Centrality'] = df['Location'].map(location_centrality_map).fillna(0)
        except Exception:
            df['Customer_Centrality'] = 0.0
            df['Location_Centrality'] = 0.0
    else:
        df['Customer_Centrality'] = 0.0
        df['Location_Centrality'] = 0.0

    # Sentiment features
    if 'Adjuster_Notes' in df.columns:
        df['Sentiment_Score'] = df['Adjuster_Notes'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
        df['Negative_Tone_Flag'] = (df['Sentiment_Score'] < -0.5).astype(int)
    else:
        df['Sentiment_Score'] = 0.0
        df['Negative_Tone_Flag'] = 0

    # Date-based features
    if 'Incident_Date' in df.columns:
        years = range(2023, 2026)
        nigerian_holidays = holidays.Nigeria(years=years)
        df['Incident_on_Holiday'] = df['Incident_Date'].apply(lambda date: int(date in nigerian_holidays) if pd.notnull(date) else 0)
        df['Incident_on_Weekend'] = df['Incident_Date'].dt.dayofweek.apply(lambda x: 1 if pd.notnull(x) and x >= 5 else 0)
    else:
        df['Incident_on_Holiday'] = 0
        df['Incident_on_Weekend'] = 0

    if 'Claim_Submission_Date' in df.columns:
        df['Claim_Submission_on_Weekend'] = df['Claim_Submission_Date'].dt.dayofweek.apply(lambda x: 1 if pd.notnull(x) and x >= 5 else 0)
    else:
        df['Claim_Submission_on_Weekend'] = 0

    if 'Claim_Submission_Date' in df.columns and 'Incident_Date' in df.columns:
        df['Days_to_Claim_Submission'] = (df['Claim_Submission_Date'] - df['Incident_Date']).dt.days.fillna(-1)
        df['Late_Claim_Submission'] = (df['Days_to_Claim_Submission'] >= 90).astype(int)
    else:
        df['Days_to_Claim_Submission'] = -1
        df['Late_Claim_Submission'] = 0

    if 'Policy_End_Date' in df.columns and 'Policy_Start_Date' in df.columns:
        df['Policy_Duration_Days'] = (df['Policy_End_Date'] - df['Policy_Start_Date']).dt.days.fillna(-1)
    else:
        df['Policy_Duration_Days'] = -1

    # Claim Count 2 Years and Frequent Claimant
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
        df['Claim_Count_2Years'] = 0
        df['Frequent_Claimant'] = 0

    # High Claim Amount Flag
    if 'Claim_Amount' in df.columns:
        percentile_90 = 454548.349
        df['High_Claim_Amount_Flag'] = (df['Claim_Amount'] > percentile_90).astype(int)
    else:
        df['High_Claim_Amount_Flag'] = 0

    # Claim vs Premium Ratio
    if 'Claim_Amount' in df.columns and 'Premium_Amount' in df.columns:
        df['Claim_vs_Premium_Ratio'] = df.apply(lambda row: row['Claim_Amount'] / row['Premium_Amount'] if row['Premium_Amount'] > 0 and pd.notnull(row['Premium_Amount']) else 0, axis=1)
    else:
        df['Claim_vs_Premium_Ratio'] = 0.0

    # Customer Claim Frequency
    if 'Customer_Name' in df.columns:
        customer_claim_counts = df.groupby('Customer_Name').size().reset_index(name='Customer_Claim_Count')
        df = df.merge(customer_claim_counts, on='Customer_Name', how='left')
        df['Frequent_Customer_Claimant'] = (df['Customer_Claim_Count'] > 2).astype(int)
    else:
        df['Customer_Claim_Count'] = 0
        df['Frequent_Customer_Claimant'] = 0

    # Prior Fraudulent Claim
    if 'Customer_Name' in df.columns and 'Prior_Fraudulent_Claim' in df.columns:
        if 'Prior_Fraudulent_Claim' not in df.columns:
            df['Prior_Fraudulent_Claim'] = 0
    else:
        df['Prior_Fraudulent_Claim'] = 0

    # Claims within 2 months of policy start/end dates
    if 'Claim_Submission_Date' in df.columns and 'Policy_Start_Date' in df.columns:
        df['Claim_Within_2Months_of_Start'] = ((df['Claim_Submission_Date'] - df['Policy_Start_Date']).dt.days <= 60).astype(int)
    else:
        df['Claim_Within_2Months_of_Start'] = 0

    if 'Claim_Submission_Date' in df.columns and 'Policy_End_Date' in df.columns:
        df['Claim_Within_2Months_of_End'] = (((df['Policy_End_Date'] - df['Claim_Submission_Date']).dt.days <= 60) &
                                             ((df['Policy_End_Date'] - df['Claim_Submission_Date']).dt.days >= 0)).astype(int)
    else:
        df['Claim_Within_2Months_of_End'] = 0

    # One-hot encoding for categorical columns
    categorical_cols_to_encode = [col for col in df.columns if df[col].dtype == 'object' and col not in date_cols + ['Customer_Name', 'Customer_Email', 'Customer_Phone', 'Adjuster_Notes', 'Policy_Number', 'Claim_ID']]
    if categorical_cols_to_encode:
        df = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)

    # Align columns with training features
    for col in features_for_splitting:
        if col not in df.columns:
            df[col] = 0

    df = df[features_for_splitting]

    # Scale numerical features
    numerical_features_original = ['Claim_Amount', 'Customer_Age', 'Premium_Amount']
    numerical_features_to_scale = [col for col in features_for_splitting if col in numerical_features_original or (df[col].dtype in ['int64', 'float64'] and not col.startswith('tfidf_') and col not in ['Customer_Centrality', 'Location_Centrality', 'Sentiment_Score', 'Negative_Tone_Flag', 'Claim_Count_2Years', 'Frequent_Claimant', 'High_Claim_Amount_Flag', 'Claim_vs_Premium_Ratio', 'Customer_Claim_Count', 'Frequent_Customer_Claimant', 'Prior_Fraudulent_Claim', 'Claim_Within_2Months_of_Start', 'Claim_Within_2Months_of_End'])]

    if numerical_features_to_scale:
        try:
            df[numerical_features_to_scale] = scaler.transform(df[numerical_features_to_scale])
        except Exception as e:
            st.warning(f"Error applying scaler: {e}. Skipping numerical scaling.")

    return df

# Streamlit App
st.title("Fraud Detection Application")
st.write("Upload your claims data (CSV format) to get fraud predictions.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.write("Original Data:")
        st.dataframe(input_df.head())
        st.write("Preprocessing data...")
        processed_df = preprocess_data(input_df.copy(), scaler, tfidf, features_for_splitting)
        st.write("Preprocessing complete. Features for prediction:")
        st.dataframe(processed_df.head())
        st.write(f"Shape of processed data: {processed_df.shape}")

        st.write("Making predictions...")
        predictions = best_model.predict(processed_df)
        predictions_proba = best_model.predict_proba(processed_df)[:, 1]

        input_df['Predicted_Fraud_Flag'] = predictions
        input_df['Predicted_Fraud_Probability'] = predictions_proba

        st.write("Predictions complete.")
        st.subheader("Prediction Results")
        cols_to_show = ['Claim_ID', 'Policy_Number', 'Customer_Name', 'Claim_Amount', 'Predicted_Fraud_Flag', 'Predicted_Fraud_Probability']
        available_cols = [col for col in cols_to_show if col in input_df.columns]
        st.dataframe(input_df[available_cols])

        fraud_count = input_df['Predicted_Fraud_Flag'].sum()
        total_claims = input_df.shape[0]
        st.write(f"Total Claims Processed: {total_claims}")
        st.write(f"Predicted Fraudulent Claims: {fraud_count}")
        st.write(f"Predicted Non-Fraudulent Claims: {total_claims - fraud_count}")

        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

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