import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
# import predictStress
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
model = pickle.load(open('xgboos.pkl', 'rb'))
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('home.html')

# SEM3
# @app.route('/sem3')
# def sem3():
#     return render_template('sem3.html')
# def predict1():
#     int_features = [float(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)

#     output = round(prediction[0], 2)

#     return render_template('sem3.html', prediction_text='student from sem 3 belongs to class : {}'.format(output))

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    user_input_reshape = final_features.to_numpy().reshape(-1, 1)
    user_input_scaled = scaler.fit_transform(user_input_reshape)
    single_row_array = user_input_scaled.reshape(1, -1)

    probability = model.predict_proba(single_row_array)
    stress = round(probability[0][1] * 100, 2)  # Probability of stress level

    print('this is the output', stress)

    return render_template('predict.html', prediction_text='student from sem 3 belongs to class : {}'.format(stress))

# @app.route('/predict_apisem3',methods=['POST'])
# def predict_apisem3():
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
    