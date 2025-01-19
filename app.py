from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import re
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

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


model_name = 'tharu0418/sentence-model'  # Or any other model you are using
#model = TFBertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Use Accelerate to load the model efficiently
print("Initializing model with Accelerate...")
with init_empty_weights():  # Load the model without weights
    model = TFBertForSequenceClassification.from_pretrained(model_name, device_map="auto")

# Distribute weights efficiently
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=model_name,
    device_map=infer_auto_device_map(model),  # Automatically infer device placement
    offload_folder="offload",  # Folder to offload layers if memory is limited
    offload_state_dict=True
)

with open('Models/label_mapping.pkl', 'rb') as f:
    label_mapping = pickle.load(f)



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



# setence model functions
# Function to predict illness
def predict_illness(text):
    # Tokenize the input text
    encoding = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="tf")
    
    # Get model output
    outputs = model(encoding)
    logits = outputs.logits
    
    # Get the predicted class index
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    
    # Map the predicted class index to the illness (intent)
    predicted_illness = label_mapping[predicted_class]
    
    return predicted_illness

@app.route('/get_sentence', methods= ['POST'])
def get_sentence():
    try:

        data = request.json
        sentence = data.get('sentence')

           # Check if the input is a string
        if not isinstance(sentence, str):
            return jsonify({"error": "Invalid sentence"}), 400
        
        # Check if the input string is empty
        if sentence.strip() == "":
            return jsonify({"error": "Please enter your sentence"}), 400

        # Predict the illness
        predicted_illness = predict_illness(sentence)
        
        # Return the result as a JSON response
        return jsonify({"symptom": predicted_illness})

        
    except Exception as e:
        return jsonify({"error" : str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)