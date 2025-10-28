import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#in case of pyplot not working
#plt.switch_backend('new_backend')

dates =[]
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader) #skips first row bc its just column names
        for row in csvFileReader:
            current_date = row[0]
            date_arr = current_date.split('-')
            if(int(date_arr[0]) == 2020):
                #dates.append(int(row[0].split('-')[2])) 
                dates.append(int(date_arr[2])) 
                prices.append(float(row[1]))
    return

def predict_prices(dates, prices, x):
    #edit this method
    #this method expects x to be a 2d array, figure out how to pass that in properly
    dates = np.reshape(dates,(len(dates), 1))
    
    svr_lin = SVR(kernel= 'linear', C=1e3)
    svr_poly = SVR(kernel = 'poly', C=1e3, degree = 2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RRF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('GME_stock.csv')

#come back to edit this
predicted_price = predict_prices(dates, prices, [[29]])

print(predicted_price)