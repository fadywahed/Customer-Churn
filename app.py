import pandas as pd
import numpy as np
import gradio as gr
import joblib

label_encoders = joblib.load("label_encoder.pkl")
le_target = joblib.load("label_target_encoder.pkl")
min_max_scaler = joblib.load("min_max_scaler.pkl")
one_hot_encoder = joblib.load("one_hot_encoder.pkl")
model = joblib.load("model.pkl")

def processed_data(data):
    df = pd.DataFrame([data])

    label_encoder_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "gender"]
    one_hot_encoder_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                       "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                        "Contract", "PaymentMethod"]
    min_max_encoder_cols = ["tenure", "MonthlyCharges", "TotalCharges"]


    for col in min_max_encoder_cols, one_hot_encoder_cols:
        df[col] = df[col].str.strip()

    df[min_max_encoder_cols] = df[min_max_encoder_cols].replace(" ", np.nan).astype(float)
    df[min_max_encoder_cols] = df[min_max_encoder_cols].fillna(df[min_max_encoder_cols].mean())


    for col in label_encoder_cols:
        le = label_encoders[col]
        df[col] = le.transform(df[col])
        
    one_hot_encoded = one_hot_encoder.transform(df[one_hot_encoder_cols])
    scaled_numrical = min_max_scaler.transform(df[min_max_encoder_cols])
    
    X_processed = np.hstack((df[label_encoder_cols].values, one_hot_encoded, scaled_numrical))
    return X_processed

def predict(gender, senior_sitizen, partner, dependents,	tenure,	phone_service, multiple_lines,	
        internet_service,	online_security,	online_backup, device_protection ,tech_support, 
        streaming_tv ,streaming_movies, contract, paperless_billing, payment_method,	monthly_charges, total_charges):
    
    
    data = {
        gender : gender,
        senior_sitizen : SeniorCitizen,
        partner : Partner,
        dependents : Dependents,
        tenure : tenure,
        phone_service : PhoneService,
        multiple_lines : MultipleLines,
        internet_service : InternetService,
        online_security : OnlineSecurity,
        online_backup : OnlineBackup,
        device_protection : DeviceProtection,
        tech_support : TechSupport,
        streaming_tv : StreamingTV,
        streaming_movies : StreamingMovies,
        contract : Contract,
        paperless_billing : PaperlessBilling,
        payment_method : PaymentMethod,
        monthly_charges : MonthlyCharges,
        total_charges : TotalCharges
    }
    
    try:
        X_new = processed_data(data)
        predication = model.predict(X_new)
        predication = le_target.transform(predication)
        return "Churn" if predication[0] == "Yes" else "No Churn"
    except Exception as e:
        print("Error during Exception", e)
        return str(e)
    
    
inputs = [
    gr.Radio(label="gender", choices=["Male","Female"]),
    gr.Number(label="Senior Citizen (0,1)"),
    gr.Radio(label="Partner", choices=["Yes","No"]),
    gr.Radio(label="Dependents", choices=["Yes","No"]),
    gr.Number(label="tenure"),
    gr.Radio(label="Phone Service", choices=["Yes","No"]),
    gr.Radio(label="Multiple Lines", choices=["Yes","No", "No phone service"]),
    gr.Radio(label="Internet Service", choices=["DSL","Fiber optic","No"]),
    gr.Radio(label="Online Security", choices=["Yes","No", "No phone service"]),
    gr.Radio(label="Online Backup", choices=["Yes","No", "No phone service"]),
    gr.Radio(label="Device Protection", choices=["Yes","No", "No phone service"]),
    gr.Radio(label="Tech Support", choices=["Yes","No", "No phone service"]),
    gr.Radio(label="Streaming TV", choices=["Yes","No", "No phone service"]),
    gr.Radio(label="Streaming Movies", choices=["Yes","No", "No phone service"]),
    gr.Radio(label="Contract", choices=["Month-to-month","One year", "Two year"]),
    gr.Radio(label="Paperless Billing", choices=["Yes","No"]),
    gr.Radio(label="Payment Method", choices=["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]),
    gr.Number(label="Monthly Charges"),
    gr.Number(label="Total Charges"),
]

outputs = gr.Textbox(label="Prediction")

gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Churn Prediction Model").lunch(share=True)