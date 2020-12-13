# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 01:36:21 2020

@author: dell
"""


import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open('classifier_titanic1.pkl', 'rb'))

def Title(d):
    x = d[0].split()
    sr = x[1]
    if sr == 'Mr.':
        return 0
    elif sr == 'Miss.':
        return 1
    elif sr == 'Mrs.':
        return 2
    elif sr == 'Master.':
        return 3
    else:
        return 4

def null_age(age):
    if ((feat['Pclass'] == 1) & (feat['Sex'] == 'male')):
        return 46
    elif ((feat['Pclass'] == 1) & (feat['Sex'] == 'female')):
        return 38
    elif ((feat['Pclass'] == 2) & (feat['Sex'] == 'male')):
        return 28
    elif ((feat['Pclass'] == 2) & (feat['Sex'] == 'female')):
        return 25
    elif ((feat['Pclass'] == 3) & (feat['Sex'] == 'male')):
        return 20
    else:
        return 17


def final_feat(feat):

    if feat['Age'].isnull()[0]:
        feat['Age'] = null_age(feat['Age'])
    
    if feat['Embarked'].isnull()[0]:
        feat['Embarked'] =='S'
    
    feat['Age'] = int(feat['Age'])
    feat['Age'] = feat['Age'].apply(lambda x: 0 if x<=15 else (1 if x<=25 else (2 if x<=45 else 3)))
    
    feat['Fare'] = int(float(feat['Fare']))
    feat['Fare'] = feat['Fare'].apply(lambda x: 0 if x<=10 else (1 if x<=50 else (2 if x<=100 else (3 if x<=200 else 4))))
    
    feat['Embarked'] = feat['Embarked'].apply(lambda x: 0 if x=='S' else (1 if x=='C' else 2))
    
    feat['Parch'] = int(feat['Parch'])
    feat['SibSp'] = int(feat['SibSp'])
    feat['TotalMembers'] = feat['Parch'] + feat['SibSp'] + 1
    
    feat['Title'] = Title(feat['Name'])
    
    feat['IsAlone'] = feat['TotalMembers'].apply(lambda x: 1 if x == 1 else 0)
    
    feat['Has_Cabin'] = feat['Cabin'].apply(lambda x: 1 if type(x) == float else 0)
    
    feat['SEX'] = feat['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    
    feat.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin'], inplace=True, axis=1)
    
    return feat


@app.route('/')
def home():
    return render_template('Titanic.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    features = [np.array(int_features)]
    feat = pd.DataFrame(data=features, columns=["PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"])
    final_features = final_feat(feat)
    prediction = model.predict(final_features)
    if prediction==0:
        return render_template('Titanic.html', prediction_text='Great this passanger has survived')
    else:
        return render_template('Titanic.html', prediction_text='Sorry this passanger had died')

if __name__ == "__main__":
    app.run(port=12000)
