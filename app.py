from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import re
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

class CustomLabelEncoder(LabelEncoder):
    def __init__(self, start=0):
        self.start = start
        super().__init__()

    def fit_transform(self, y):
        encoded = super().fit_transform(y)
        encoded += self.start
        return encoded


    
app = Flask(__name__)
CORS(app)


def getSymptomCSV():
    global df_symptoms
    df_symptoms = pd.read_csv('DB/Symptoms.csv')
def getIllnessCSV():
    global df_illness
    df_illness = pd.read_csv('DB/Illness.csv')
def getIllnessDesription():
    global df_diseases_list
    df_diseases_list = pd.read_csv('DB/symptom_Description.csv')

def getPreventionDetails():
    global df_preventions
    df_preventions = pd.read_csv('DB/symptom_precaution.csv',  encoding='ISO-8859-1')

def getTreatmentDetails():
    global df_treatments
    df_treatments = pd.read_csv('DB/symptom_treatments.csv', encoding='ISO-8859-1')


# Load the saved models and encoders **after** defining the class
with open("Models/rf_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

with open("Models/label_encoder.pkl", "rb") as le_file:
    encoder = pickle.load(le_file)

with open("Models/mlb.pkl", "rb") as mlb_file:
    mlb = pickle.load(mlb_file)


def strip_to_basic_tokens(text):
    # Remove double spaces and underscores, then split by commas and lowercase the tokens
    text = re.sub(r'[_\s]+', ' ', text)
    tokens = [token.strip().lower() for token in text.split(',')]
    return tokens

    

# Get diseases names based on symptoms
@app.route('/get_diseases_name', methods= ['POST'])
def get_diseases_name():
    try:
        getSymptomCSV()

        data = request.json
        symptom_input = data['symptom_user_input']

        matching_diseases = df_symptoms[df_symptoms['symptom'] == symptom_input]

        # Check if matching_diseases is empty
        if matching_diseases.empty:
            return jsonify({"message": "Enter correct symptom"}), 400


        matching_diseases = matching_diseases.iloc[0, 1:].dropna().unique()

        return jsonify({"diseases" : matching_diseases.tolist()})


    except Exception as e:
        return jsonify({"error" : str(e)}), 500

# Get all symptoms names based on diseases
@app.route('/get_illess_name', methods = ['POST'])
def get_illess_name():
    try:
        getIllnessCSV()
        data = request.json
        diseases_input = data['diseases_input']

        print('diseases_input', diseases_input)

        matching_symptoms = df_illness[df_illness['Disease'] == diseases_input]
        matching_symptoms = matching_symptoms.iloc[0, 1:].dropna().unique()

        return jsonify({"symptoms" : matching_symptoms.tolist()})

    except Exception as e:
        return jsonify({"error" : str(e)}), 500


# Predict user's diseases
@app.route('/predictthediseases', methods = ['POST'])
def predictthediseases():
    try:

        data = request.json
        symptoms =  data.get('symptoms', "")

        print('symptoms', symptoms)
        
        if not symptoms:
            return jsonify({"error": "No symptoms provided."}), 400


        false_return = 'Not_Available'
        no_null_count = len([s for s in symptoms if s])

        # If user select less than 1 symptom return 'NO'
        if no_null_count <= 1:
            return jsonify({"predicted_illness" : false_return})

        basic_tokens = strip_to_basic_tokens(symptoms)

        one_hot_encoded_sample = mlb.transform([basic_tokens])

        one_hot_df = pd.DataFrame(one_hot_encoded_sample, columns=mlb.classes_)

        missing_columns = set(mlb.classes_) - set(one_hot_df.columns)
        for col in missing_columns:
            one_hot_df[col] = 0

        one_hot_df = one_hot_df[mlb.classes_]

        y_pred = rf_model.predict(one_hot_df)

        if not y_pred.any():
            return jsonify({"predicted_disease": "No_Matching"}), 200

        predicted_class_index = np.argmax(y_pred)
        predicted_disease = encoder.inverse_transform([predicted_class_index])[0]

        return jsonify({"predicted_disease": predicted_disease}), 200


    except Exception as e:
        return jsonify({"error" : str(e)}), 500
    
# Get illness preventions and treatments
@app.route('/getpreventions', methods=['POST'])
def getPreventions():
    try:
        getIllnessDesription()
        getPreventionDetails()
        getTreatmentDetails()

        data = request.json
        diseases = data.get('diseases')

        discription = df_diseases_list[df_diseases_list['Disease'] == diseases]
        discription = discription.values[0][1]

        print('discription', discription)

        prevention_list = []
        treatment_list = []

        prevention_row = df_preventions[df_preventions['Disease'] == diseases]

        treatment_row = df_treatments[df_treatments['Disease'] == diseases]


        if not prevention_row.empty:
            for i in range(1, len(prevention_row.columns)):
                prevention = prevention_row.iloc[0, i]
                if pd.notna(prevention):
                    prevention_list.append(prevention)


        if not treatment_row.empty:
            for i in range(1, len(treatment_row.columns)):
                treatment = treatment_row.iloc[0, i]
                if pd.notna(treatment):
                    treatment_list.append(treatment)

        


        return jsonify({"description" : discription, "prevntion_list" : prevention_list, "treatment_list" : treatment_list})


    except Exception as e:
        return jsonify({"error" : str(e)}), 500    




if __name__ == '__main__':
    app.run(debug=True)