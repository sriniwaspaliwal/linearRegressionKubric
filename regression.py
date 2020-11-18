import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE

    #This function returns the variance of val    
    def var(val, m):
        s = 0
        for i in val:
            s += (i-m)**2
        return s
    
    #This function returns the covariance of x and y 
    def covar(x, m_x, y, m_y):
    	co = 0.0
    	for i in range(len(x)):
    		co += (x[i] - m_x) * (y[i] - m_y)
    	return co
    
    
    data = response.content.decode("utf-8")
    ar, price = data.split("\n")[0:2]
    ar = list(map(float, ar.split(",")[1:]))
    price = list(map(float, price.split(",")[1:]))
    
    m_x, m_y = numpy.mean(ar), numpy.mean(price)
    var_x = var(ar, m_x)
    co = covar(ar, m_x, price, m_y)
    b1 = co / var_x
    b0 = m_y - b1 * m_x
    
    p = []
    for i in area:
        temp = b0 + b1*i
        p.append(temp)
    return p
    ...


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
