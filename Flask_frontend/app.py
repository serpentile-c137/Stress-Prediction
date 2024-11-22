import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
model = pickle.load(open(r'new_logistic_regression_model.pkl', 'rb'))
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/tryit')
def tryit():
    return render_template('tryit.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/predict',methods=['POST'])
def predict():
    #int_features = [[float(x) for x in request.form.values()]]
    EDA = request.form.get("EDA")
    TEMP = request.form.get("TEMP")
    BVP = request.form.get("BVP")
    HR = request.form.get("HR")

    user_input = pd.DataFrame([[EDA, TEMP, BVP, HR]], columns=['EDA', 'TEMP', 'BVP','HR'])
 
# Scale the input data
    user_input_reshape = user_input.to_numpy().reshape(-1, 1)
    user_input_scaled = scaler.fit_transform(user_input_reshape)
    single_row_array = user_input_scaled.reshape(1, -1)

    probability = model.predict_proba(single_row_array)
    # stress = round(probability[0][1], 4)
    stress = round(probability[0][1] * 100, 2)  # Probability of stress level

    print('this is the output', stress)

    # return render_template('predict.html', prediction_text='stress prboablity is : {}'.format(stress))
    return render_template('predict.html', prediction_text='{}'.format(stress))

if __name__ == "__main__":
    app.run(debug=True)
    