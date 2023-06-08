from django.shortcuts import render

import pandas as pd
import matplotlib.pyplot as olt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    data = pd.read_csv(r"C:\Users\ADMIN\Desktop\KPDL\BTL\diabetes.csv")
    X = data.drop("Outcome", axis=1)
    Y = data['Outcome']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, Y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    result2 = ""
    if pred == [1] and val2 > 130:
        result2 = " Tiểu đường"
        advice = " Ăn uống theo chế độ, giảm tinh bột, ưu tiên ăn chay, bớt ăn đồ ngọt"
    elif  pred == [1] and val2 >= 110 and val2 <= 130:
        result2 = " Tiền tiểu đường"
        advice = " Cần ăn uống điều độ, giảm tinh bột, ăn nhiều rau củ ít tinh bột"
    else:
        result2 = " Bình thường"
        advice = " Không bị bệnh, nên giữ chế độ ăn uống hiện tại và healthy hơn nếu có thể"

    return render(request, 'predict.html', {"result2": result2, "advice": advice})