from flask import Flask, render_template, request

from model import load_data, train_models, predict_models

app = Flask(__name__)

# Load data and train models
X, y = load_data()
trained_models, model_accuracies = train_models(X, y)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from the form
    data = {
        'Sex': request.form['Sex'],
        'GeneralHealth': request.form['GeneralHealth'],
        'AgeCategory': request.form['AgeCategory'],
        'HeightInMeters': float(request.form['Height']),
        'WeightInKilograms': float(request.form['Weight']),
        'BMI': float(request.form['BMI']),
        'HadAngina': int(request.form.get('HadAngina', 0)),
        'HadStroke': int(request.form.get('HadStroke', 0)),
        'HadDepressiveDisorder': int(request.form.get('HadDepressiveDisorder', 0)),
        #'HadKidneyDisease': int(request.form.get('HadKidneyDisease', 0)),
        'HadDiabetes': request.form['HadDiabetes'],
        'SmokerStatus': request.form['SmokerStatus'],
        'ECigaretteUsage': request.form['ECigaretteUsage'],
        #'ChestScan': int(request.form.get('ChestScan', 0)),
        #'RaceEthnicityCategory': request.form['RaceEthnicityCategory'],
        'AlcoholDrinkers': request.form['AlcoholDrinkers'],
        'CovidPos': int(request.form.get('CovidPos', 0))
    }

    prediction_results = predict_models(data,trained_models,model_accuracies)
    #prediction_message = {model: result for model, result in prediction_results.items()}
    return render_template('result.html', model_accuracies=model_accuracies, prediction=prediction_results)


if __name__ == '__main__':
    app.run(debug=True)
