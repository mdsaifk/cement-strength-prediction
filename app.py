from flask import Flask, render_template, jsonify,request
import pickle
from wsgiref import simple_server
import os
app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        Cement_component_1 = float(request.form['Cement_component_1'])

        Blast_Furnace_Slag_component_2 = float(request.form['Blast_Furnace_Slag_component_2'])

        Fly_Ash_component_3 = float(request.form['Fly_Ash_component_3'])

        Water_component_4 = float(request.form['Water_component_4'])

        Superplasticizer_component_5 = float(request.form['Superplasticizer_component_5'])

        Coarse_Aggregate_component_6 = float(request.form['Coarse_Aggregate_component_6'])

        Fine_Aggregate_component_7 = float(request.form['Fine_Aggregate_component_7'])

        Age_day = float(request.form['Age_day'])

        standardized_data = pickle.load(open('standard_scaler.pkl', 'rb'))
        model = pickle.load(open('randomforest.pkl', 'rb'))

        prediction= model.predict(standardized_data.transform([[Cement_component_1,Blast_Furnace_Slag_component_2,Fly_Ash_component_3,Water_component_4,Superplasticizer_component_5,Coarse_Aggregate_component_6,Fine_Aggregate_component_7,Age_day]]))
        output=round(prediction[0],2)

        if output>0:
            return render_template('index.html', prediction_text=f'The strength of the concrete will be : {output}')
        else:
            return render_template('index.html',prediction_text=f'The concrete strength is : {output}. Strength is very bad you need to check the components')
#
#
#
if __name__ == "__main__":
    port = int(os.getenv("PORT"))
    app = app()
    host = '0.0.0.0'
    httpd = simple_server.make_server(host=host,port=port, app=app)
    httpd.serve_forever()