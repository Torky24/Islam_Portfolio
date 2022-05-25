import numpy 
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_excel('/Users/islamtorky/Desktop/transportation regression/i torky.xlsx')
X = df.iloc[:,0:5].values
y = df.iloc[:, 5].values

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 120, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(X)

#Characteristics of Regressor
r2 = r2_score(y,y_pred)
rmse = numpy.sqrt(mean_squared_error(y,y_pred))


#________________________________________________________________________________________________________________
#Drawing P-I Diagram
TR = 100 #Initial Prediction Trials

WALLNO = 3
RU_Ktemp = X[WALLNO,2:4].reshape(1,1,2)
RU_K = numpy.zeros((TR,2))

for t in range(TR): #Filling numpy with same value TR times
    RU_K[t] = RU_Ktemp
      

#Filling Pressure With Random Values, and Scaling them
Ptemp = numpy.random.randint(10000, size=(TR,1))
P = Ptemp.reshape((TR,1))


#Filling Impulse With Random Values, and Scaling them
Itemp = numpy.random.randint(10000, size=(TR,1))
I = Itemp.reshape((TR,1))


#Invidual Values are ready
P_I = numpy.append(P,I, axis = 1)

RKPI = numpy.append(P_I, RU_K , axis=1)

b_s = numpy.array([regressor.predict(RKPI)]).T


KABOOM = numpy.append(RKPI, b_s,axis =1) 

#________________________________________________________________________________________________________________

#Start Loop to find adequate amount of combined P-I Values to plot

LOD = 3.25 #Damage Criteria in degrees

cc = pd.DataFrame()

for t in range (TR):
    if KABOOM[:,4][t] == (LOD):
        cc[t] = KABOOM[t]


cc = cc.T
xx = len(cc.columns)
print(xx)

while xx < 100:
   
    Ptemp = numpy.random.randint(low = 99, high = 230 , size=(TR,1))
    P = Ptemp.reshape((TR,1))
    Itemp = numpy.random.randint(low = 0, high = 100 , size=(TR,1))
    I = Itemp.reshape((TR,1))
    P_I = numpy.append(P,I, axis = 1)


    RKPI = numpy.append(P_I, RU_K , axis=1)

    b_s = numpy.array([regressor.predict(RKPI)]).T
    KABOOM = numpy.append(RKPI, b_s , axis=1) 
    print(KABOOM[10,4])

    for t in range (TR):
        if KABOOM[:,4][t] >= (LOD - 0.05) and KABOOM[:,4][t] <= (LOD + 0.05):
            cc[t] = KABOOM[t]
    
    xx = len(cc.columns)
    if xx == TR:
        break

PP = cc.iloc[0,:]
II = cc.iloc[1,:]

plt.scatter(II,PP)
