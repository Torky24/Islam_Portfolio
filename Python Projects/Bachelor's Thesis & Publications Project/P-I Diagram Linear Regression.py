import numpy as np
import pandas as pd

from pylab import rcParams
import seaborn as sb


import sklearn as mdf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error



rcParams['figure.figsize']=6,4
sb.set_style('darkgrid')

database = '/Users/islamtorky/Desktop/transportation regression/i torky.xlsx'
df = pd.read_excel(database)
mdf.columns = ['pres','imp','ru','k','vr rft','theta']

sb.pairplot(df)

df_data=df.ix[:,:5].values
df_target = df.ix[:,5].values

X,y = scale(df_data), df_target



LinReg = LinearRegression(normalize = True)
LinReg.fit(X,y)

y_pred = LinReg.predict(X)


print('predicted response:', y_pred, sep='\n')

r_sq = LinReg.score(X, y)
print('coefficient of determination:', r_sq)

rmse = np.sqrt(mean_squared_error(y,y_pred))

print('slope:', LinReg.coef_)

print('intercept:', LinReg.intercept_)

print(LinReg.score(X,y))
