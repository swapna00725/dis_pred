from flask import Flask, request, render_template
import os
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        symptoms = request.form["symptoms"]

        # ✅ Create pipeline instance
        pipeline = PredictPipeline()
        rf_pred, knn_pred, dt_pred, final_pred = pipeline.predict_disease(symptoms)

        return render_template(
            "result.html",
            symptoms=symptoms,
            rf_pred=rf_pred,
            knn_pred=knn_pred,
            dt_pred=dt_pred,
            final_pred=final_pred
        )  
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)