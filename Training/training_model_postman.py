from flask import Flask, render_template, jsonify,request
import pickle
from Training_model_tunetrain_module.model_tuning_and_training import tuning_training

app = Flask(__name__)
@app.route('/training',methods=['POST'])
def training_model():
    if (request.method =='POST'):
        operation=request.json['operation']

        if (operation.lower() == 'training'):

            model_object=tuning_training()
            result=model_object.tune_train()

            return jsonify(f'Result:{result}')

if __name__=='__main__':
    app.run(port=5000, debug=True)