from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

app = Flask(__name__)
CORS(app)


model_name = 'tharu0418/sentence-model'  # Or any other model you are using
model = TFBertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

with open('label_mapping.pkl', 'rb') as f:
    label_mapping = pickle.load(f)


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


########################################







if __name__ == '__main__':
    app.run(debug=True)