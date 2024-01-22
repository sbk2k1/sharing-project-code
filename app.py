# data inputs required for prediction

# Gender (0/1)
# Married (0/1)
# Dependents (0/1/2/3+)
# Education (0/1)
# Self_Employed (0/1)
# ApplicantIncome (int)
# CoapplicantIncome (int)
# LoanAmount (int)
# Loan_Amount_Term (int)
# Credit_History (0/1)
# Property_Area (0/1/2)

# inputs that need scaling (robust scaler)

# ApplicantIncome (int)
# CoapplicantIncome (int)
# LoanAmount (int)
# Loan_Amount_Term (int)
# Credit_History (0/1)

# code required to predict the output

# pickle.dump(Model8, open("logistic_model_by_rj.pkl", 'wb'))
# imported_model = pickle.load(open("logistic_model_by_rj.pkl", 'rb'))
# y_pred_after_load = imported_model.predict(X_test)
# print(classification_report(y_pred_after_load,y_test))

# we need to import pickle for the model, sklearn for preprocessing and flask for the app
import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
# for scaling we need to import RobustScaler
from sklearn.preprocessing import RobustScaler

# we need to import the model
imported_model = pickle.load(open("logistic_model_by_rj.pkl", 'rb'))

# we need to import the scaler
scaler = pickle.load(open("robust_scaler_by_rj.pkl", 'rb'))

#start the flask app
app = Flask(__name__)

# create the route for the app
@app.route('/')
def home():
    return render_template('index.html')

# create the route for the prediction
@app.route('/predict', methods=['POST'])
def predict():



    # get the data from the form
    user_input = request.form.values()
    # convert the list into a numpy array
    user_input_array = np.array(list(user_input))
    # create a dataframe
    user_input_df = pd.DataFrame(user_input_array.reshape(1,-1))
    # prepare the data for prediction
    # we need to scale the data
    # we need to convert the data into float
    user_input_df = user_input_df.astype(float)




    # we need to scale the data
    # we need to import the scaler
    user_input_df[[1,2,3,4,5]] = scaler.transform(user_input_df[[1,2,3,4,5]])
    # remove column 5 since it was needed only for scaling
    user_input_df = user_input_df.drop(columns=[4])


    # we need to predict the output
    prediction = imported_model.predict(user_input_df)
    # we need to convert the prediction into a string
    prediction_string = str(prediction)


    
    # we need to return the prediction
    if prediction_string == '[1]':
        return render_template('result.html', prediction_text='Your loan status is Approved')
    else:
        return render_template('result.html', prediction_text='Your loan status is fucked')

# run the app
if __name__ == '__main__':
    app.run(debug=True)

