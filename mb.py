import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import tensorflow as tf
from statistics import mode




#Initialize the flask App
app = Flask(__name__)

model = pickle.load(open('models/model_lr.pkl', 'rb'))
model1 = pickle.load(open('models/model_rf.pkl', 'rb'))
model2 = tf.keras.models.load_model("models/model") # It can be used to reconstruct the model identically.

X_train = pickle.load(open('models/X_train.pkl','rb'))


scx = pickle.load(open('models/scx.pkl','rb')) #minmax scaler
onec = pickle.load(open('models/onec.pkl','rb'))
X_train = pickle.load(open('models/X_train.pkl','rb'))
X_train_cat = pickle.load(open('models/X_train_cat.pkl','rb'))
y_train = pickle.load(open('models/y_train.pkl','rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    cat_features = features[0:5]
    num_features = features[5:10]
    X_test = {'JobRole':[cat_features[0]],'Department':[cat_features[1]],'BusinessTravel':[cat_features[2]],'EducationField':[cat_features[3]],'MaritalStatus':[cat_features[4]]}
    X_test = pd.DataFrame(X_test)
    onec_x = onec.fit(X_train_cat)  #fitting the encoder object onto the train dataset
    X_test = onec.transform(X_test) #converting the categorical features intop ordinal values based on the fit on training dataset
    X_test = pd.DataFrame(X_test)
    col_one_list = X_test.values.tolist()
    cat_features = col_one_list[0]
    final_features = [*cat_features,*num_features] #combining two lists 
    
    final_features = [np.array(final_features)]
    x = scx.fit(X_train)
    final_features = scx.transform(final_features)
    prediction = model.predict(final_features)
    prediction_1 = model1.predict(final_features)
    prediction_2 = model2.predict(final_features)
    output = prediction.tolist()
    output1 = prediction_1.tolist()
    output2 = prediction_2.tolist()
    ls = [output[0],output1[0],int(output2[0][0])]
    
    result = mode(ls) # Returns the highest occurring item
    if result == 1:
        rs = "Employee is likely to leave"
    else:
        rs = "Employee is not likely to leave"

    return render_template('index.html', prediction_text=rs)


if __name__ == "__main__":
    app.run(debug=True)
