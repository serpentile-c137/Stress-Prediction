import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
# import predictStress
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
model = pickle.load(open('new_logistic_regression_model.pkl', 'rb'))
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/tryit')
def tryit():
    return render_template('tryit.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [[float(x) for x in request.form.values()]]
    print(int_features)
    # final_features = [np.array(int_features)]
    # user_input_reshape = int_features.to_numpy().reshape(-1, 1)
    user_input_scaled = scaler.fit_transform(int_features)
    print(user_input_scaled)
    single_row_array = user_input_scaled.reshape(1, -1)
    print(single_row_array)

    probability = model.predict_proba(single_row_array)
    stress = round(probability[0][1] * 100, 2)  # Probability of stress level

    print('this is the output', stress)

    return render_template('predict.html', prediction_text='stress prboablity is : {}'.format(stress))

if __name__ == "__main__":
    app.run(debug=True)
    