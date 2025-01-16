from flask import Flask, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)










# Get diseases names based on symptoms
@app.route('/get_diseases_name', methods= ['POST'])
def get_diseases_name():
    try:
        
        # Check if matching_diseases is empty
        if matching_diseases.empty:
            return jsonify({"message": "Enter correct symptom"}), 400


        matching_diseases = matching_diseases.iloc[0, 1:].dropna().unique()

        return jsonify({"diseases" : matching_diseases.tolist()})


    except Exception as e:
        return jsonify({"error" : str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)