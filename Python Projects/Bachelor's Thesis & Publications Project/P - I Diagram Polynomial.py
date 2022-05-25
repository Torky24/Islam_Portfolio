import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pylab import rcParams
import seaborn as sb


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


database = '/Users/islamtorky/Desktop/transportation regression/i torky.xlsx'
df = pd.read_excel(database)

x = df.ix[:,:5].values #input
y = df.ix[:,5].values  #output

rcParams['figure.figsize']=6,4
sb.set_style('darkgrid')

polynomial_features= PolynomialFeatures(degree=100) #determining degree of polynomial
x_poly = polynomial_features.fit_transform(x) #fitting data

#altering data to become polynomial
model = LinearRegression()
model.fit(x_poly, y)
f = model.fit(x_poly, y)
results = model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

#results
rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
slope = model.coef_
intercept = f.intercept_


print(r2)
print(rmse)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
