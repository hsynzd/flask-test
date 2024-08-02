from flask import Flask, request, jsonify
import joblib
import logging
import pandas as pd
import random
from datetime import datetime
import numpy as np
import pickle
from custom_encoder import CustomLabelEncoder
import os
from waitress import serve

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

# get pkl datas
model_path = os.path.join(os.getcwd(), 'models', 'xgb_model.pkl')
scaler_path = os.path.join(os.getcwd(), 'models', 'scaler.pkl')
label_encoder_path = os.path.join(os.getcwd(), 'models', 'label_encoders.pkl')

# Model ve diğer dosyaları yükleyin
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)


#label_encoder = joblib.load("C:/Users/Jeyhun/Desktop/pkl models/label_encoders.pkl")

categories = ['misc_net', 'grocery_pos', 'entertainment', 'gas_transport' ,'misc_pos','grocery_net', 'shopping_net' ,'shopping_pos', 'food_dining', 'personal_care','health_fitness' ,'travel' ,'kids_pets', 'home']
genders = ['gender_F', 'gender_M']
states = ['state_AK', 'state_AL', 'state_AR', 'state_AZ', 'state_CA', 'state_CO',
       'state_CT', 'state_DC', 'state_DE', 'state_FL', 'state_GA', 'state_HI',
       'state_IA', 'state_ID', 'state_IL', 'state_IN', 'state_KS', 'state_KY',
       'state_LA', 'state_MA', 'state_MD', 'state_ME', 'state_MI', 'state_MN',
       'state_MO', 'state_MS', 'state_MT', 'state_NC', 'state_ND', 'state_NE',
       'state_NH', 'state_NJ', 'state_NM', 'state_NV', 'state_NY', 'state_OH',
       'state_OK', 'state_OR', 'state_PA', 'state_RI', 'state_SC', 'state_SD',
       'state_TN', 'state_TX', 'state_UT', 'state_VA', 'state_VT', 'state_WA',
       'state_WI', 'state_WV', 'state_WY']


day_of_week_mapping = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

custom_label_encoder_merchant = CustomLabelEncoder()
custom_label_encoder_first = CustomLabelEncoder()
custom_label_encoder_last = CustomLabelEncoder()
custom_label_encoder_job = CustomLabelEncoder()


def preprocess_data(data):
    
    try:
        df = pd.DataFrame(data , index=[0])
        required_columns = ['merchant','category','trans_date_trans_time',"cc_num",'first',
                            'last','amt','city_pop','gender','state','zip','lat','long','job','dob', "merch_lat" ,"merch_long"] 
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        df = df[required_columns] 
        
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['dob'] = pd.to_datetime(df['dob'])
    # Extract features from the datetime column
    
        df['year'] = df['trans_date_trans_time'].dt.year
        df['month'] = df['trans_date_trans_time'].dt.month
        df['day'] = df['trans_date_trans_time'].dt.day
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.day_name()
    # Map the day of the week to numeric values  
      
        df['day_of_week_num'] = df['day_of_week'].map(day_of_week_mapping)
        
        df['age'] = ((datetime.now() - df['dob']).dt.days / 365.25).astype(int)
        
        df = df.drop(columns=['trans_date_trans_time'])
        df = df.drop(columns=['dob'])
        df = df.drop(columns=['day_of_week'])
        
        df['merchant'] = df['merchant'].str.replace('fraud_', '')
        
        for i in range(len(df)):
            if df['category'].iloc[i] not in categories:
                df.at[i, 'category'] = random.choice(categories)
            
            if df['gender'].iloc[i] not in genders:
                df.at[i, 'gender'] = random.choice(genders)
                
            if df['state'].iloc[i] not in states:
                df.at[i, 'state'] = random.choice(states)
        
        encoded_columns = pd.get_dummies(df[['category', 'gender', 'state']])
        
        for col in categories:
            if col not in encoded_columns.columns:
                encoded_columns[col] = 0
        for col in genders:
            if col not in encoded_columns.columns:
                encoded_columns[col] = 0
        for col in states:
            if col not in encoded_columns.columns:
                encoded_columns[col] = 0 
                
        df = pd.concat([df.drop(columns=['category', 'gender', 'state']), encoded_columns], axis=1)
        
        df['merchant'] = label_encoder['merchant_encoder'].transform(df['merchant'])
        df['first'] = label_encoder['first_encoder'].transform(df['first'])
        df['last'] = label_encoder['last_encoder'].transform(df['last'])
        df['job'] = label_encoder['job_encoder'].transform(df['job'])
        
        
        expected_columns = scaler.feature_names_in_  # Or use the columns you have from your scaler fitting phase
        df = df.reindex(columns=expected_columns, fill_value=0)
        
        df_scaled = scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
        
        df_scaled_json = df_scaled.to_dict(orient='records')
        
        return True, df_scaled_json

    except Exception as e:
        app.logger.error(f"Data preprocessing error: {e}")
        return False, "Internal Server Error"
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        success, result = preprocess_data(data)
        
        if not success:
            return jsonify({'error': result}), 400
        
        processed_data = pd.DataFrame(result) 
        prediction = model.predict(processed_data)
        return jsonify({'prediction': prediction.tolist()}), 200
    
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500 
