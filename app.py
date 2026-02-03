from flask import Flask ,request ,jsonify ,render_template
import pandas as pd
import pickle

app = Flask(__name__)

def get_cleaned_data(form_data):
    gestation = float(form_data['gestation'])
    parity = int(form_data['parity'])
    age = float(form_data['age'])
    height = float(form_data['height'])
    weight = float(form_data['weight'])
    smoke = float(form_data['smoke'])

    cleaned_data = {"gestation":[gestation],
                    "parity":[parity],
                    "age":[age],
                    "height":[height],
                    "weight":[weight],
                    "smoke":[smoke]
                    }
    return cleaned_data

#define endpoint
# 1. होम रूट जोड़ें ताकि index.html दिखे
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')
 
@app.route("/predict", methods=['POST'])
def get_prediction():

     # get data from user
    baby_data_form = request.form

    baby_data_cleaned = get_cleaned_data(baby_data_form)

    # convert into dataframe
    baby_df = pd.DataFrame(baby_data_cleaned)

    #load ml trained model
    with open("model.pkl",'rb') as obj:
        my_model = pickle.load(obj)

    #make prediction on user data
    prediction = my_model.predict(baby_df)
    prediction = round(float(prediction[0]),2)

    #return response in a json format
    response = {"Prediction":prediction}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

""" from flask import Flask, request, jsonify ,render_template
import pandas as pd
import pickle

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

# मॉडल को यहाँ लोड करें ताकि यह एक बार लोड हो
with open("model/model.pkl", "rb") as obj:
    my_model = pickle.load(obj)

@app.route("/predict", methods=["POST"])
def get_prediction():
    try:
        # 1. यूजर से JSON डेटा प्राप्त करें
        baby_data = request.get_json() 
        
        # 2. इसे DataFrame में बदलें (Square brackets [] लगाना जरूरी है)
        # यह आपके JSON को नोटबुक वाले 'test_df' जैसा बना देगा
        baby_df = pd.DataFrame([baby_data])
        
        # 3. प्रेडिक्शन करें
        # Notebook में रिजल्ट array([126.704...]) आ रहा है
        prediction_raw = my_model.predict(baby_df)
        
        # 4. एरे से नंबर बाहर निकालें और राउंड करें
        prediction = round(float(prediction_raw[0]), 2)
        
        return jsonify({"Prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True) """