from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle,joblib

app=Flask(__name__)


## Route for home page
@app.route('/')
def home():
    return render_template('home.html') 

# 🚀 Prediction route

@app.route('/predict', methods=['POST'])

def predict():
    fever=request.form.get('fever'),
    headache=request.form.get('headache'),
    nausea=request.form.get('nausea'),
    vomiting=request.form.get('vomiting'),
    fatigue=request.form.get('fatigue'),
    joint_pain=request.form.get('joint_pain'),
    skin_rash=request.form.get('skin_rash'),
    cough=request.form.get('cough'),
    weight_loss=request.form.get('weight_loss'),
    yellow_eyes=request.form.get('yellow_eyes') 
            
# 👩‍🔬 Convert input into DataFrame
 
    input_data= pd.DataFrame({
                  'fever': [fever],
                  'headache': [headache],
                  'nausea':[nausea],
                  'vomiting': [vomiting],
                  'fatigue':[fatigue],
                  'joint_pain':[joint_pain],
                  'skin_rash':[skin_rash],
                  'cough':[cough],
                  'weight_loss':[weight_loss],
                  'yellow_eyes':[yellow_eyes]
})
    
    loaded_model = joblib.load('dis_model1.joblib')
    pred = loaded_model.predict(input_data)[0]
            #result = "dinner" if pred == 0 else "lunch
    return render_template('home.html', prediction=pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000, debug=True)