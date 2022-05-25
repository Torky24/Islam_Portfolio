# IMPORT LIBRARIES
import numpy
import matplotlib.pyplot as plt
import math
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#Load the X dataset and Y dataset
df = pd.read_excel('/Users/islamtorky/Desktop/Neural Networks/Islam torky shear wall data.xlsx')
# Check your data
print(df.head(10))

# Distribute the data to input and output
dataframeX = df.iloc[:,:5].values
dataframe = df['Theta'].values
# Create initial datasets
datasetX = numpy.reshape(dataframeX, (dataframeX.shape[0], dataframeX.shape[1]))
dataset = numpy.reshape(dataframe, (dataframe.shape[0], 1))
datasetX = datasetX.astype('float32')
dataset = dataset.astype('float32')

#Normalize the datasetX

#scalerX = MinMaxScaler(feature_range=(0, 1))
#datasetX_s = scalerX.fit_transform(datasetX)
datasetX_s = datasetX


# normalize the datasetX
scalerX1 = MinMaxScaler(feature_range=(0, 1))
temp = datasetX[:,0]
temp = numpy.reshape(temp, (temp.shape[0],1))
datasetX_s[:,0] = (scalerX1.fit_transform(temp)).reshape(temp.shape[0])
scalerX2 = MinMaxScaler(feature_range=(0, 1))
temp = datasetX[:,1]
temp = numpy.reshape(temp, (temp.shape[0],1))
datasetX_s[:,1] = (scalerX2.fit_transform(temp)).reshape(temp.shape[0])
scalerX3 = MinMaxScaler(feature_range=(0, 1))
temp = datasetX[:,2]
temp = numpy.reshape(temp, (temp.shape[0],1))
datasetX_s[:,2] = (scalerX3.fit_transform(temp)).reshape(temp.shape[0])
scalerX4 = MinMaxScaler(feature_range=(0, 1))
temp = datasetX[:,3]
temp = numpy.reshape(temp, (temp.shape[0],1))
datasetX_s[:,3] = (scalerX4.fit_transform(temp)).reshape(temp.shape[0])
scalerX5 = MinMaxScaler(feature_range=(0, 1))
temp = datasetX[:,4]
temp = numpy.reshape(temp, (temp.shape[0],1))
datasetX_s[:,4] = (scalerX5.fit_transform(temp)).reshape(temp.shape[0])

# normalize the dataset output
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_s = scaler.fit_transform(dataset)

# split into train and test sets
train_sizeX = int(len(datasetX) * 0.80)
test_sizeX = len(datasetX) - train_sizeX
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
# distribute data into train and test sets
trainX, testX = datasetX_s[0:train_sizeX,:], datasetX_s[train_sizeX:len(datasetX),:]
trainY, testY = dataset_s[0:train_size,:], dataset_s[train_size:len(dataset),:]
# Input and output dimensions
input_features = datasetX_s.shape[1]
output_dim = dataset_s.shape[1]
# Reshape model inputs
trainX = numpy.reshape(trainX, (trainX.shape[0],1,trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0],1,testX.shape[1]))


# clears any old training
tf.keras.backend.clear_session()


# Create and fit the DNN network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(5, activation='tanh', input_shape=(None, input_features)))
model.add(tf.keras.layers.Dense(31, activation='tanh')) #you can use either 'tanh','sigmoid','relu'
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='tanh'))
model.add(tf.keras.layers.Dense(7, activation='relu'))
#model.add(tf.keras.layers.Dense(5, activation='tanh'))
model.add(tf.keras.layers.Dense(output_dim))
model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=["mae"])
concretemodel = model.fit(trainX, trainY, epochs=1000, batch_size=100, 
                          validation_split = 0.3, 
                          shuffle = True, verbose=1)



Bestmodel22 = tf.keras.models.load_model(
    '/Users/islamtorky/Desktop/Neural Networks/Post-Disc4.h5',
    custom_objects=None,
    compile=True
)

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# Print 1 predictions
cn = 8
print(datasetX[cn,:])
Xnew = datasetX[cn,:]
Xnew_s = datasetX[cn,:]
#print(scalerX1.fit_transform(Xnew.reshape(-1,1)))
Xnew_s[0] = scalerX1.fit_transform(Xnew[0].reshape(-1,1))
#print(scalerX2.fit_transform(Xnew.reshape(-1,1)))
Xnew_s[1] = scalerX2.fit_transform(Xnew[1].reshape(-1,1))
#print(scalerX3.fit_transform(Xnew.reshape(-1,1)))
Xnew_s[2] = scalerX3.fit_transform(Xnew[2].reshape(-1,1))
#print(scalerX4.fit_transform(Xnew.reshape(-1,1)))
Xnew_s[3] = scalerX4.fit_transform(Xnew[3].reshape(-1,1))
print(model.predict(Xnew_s.reshape(1,1,input_features)))
Ynew_s = model.predict(Xnew_s.reshape(1,1,input_features))
Ynew = scaler.inverse_transform(Ynew_s.reshape(-1,1))
print(numpy.str(Ynew[0,0]) , "Degrees - Predicted")
print(dataset[cn,0] , "Degrees - True")

# reshape output to be [samples, features]
trainPredict = numpy.reshape(trainPredict, (trainPredict.shape[0],trainPredict.shape[2]))
testPredict = numpy.reshape(testPredict, (testPredict.shape[0],testPredict.shape[2]))
# Invert predictions(descale)
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(dataset[0:train_size,:], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(dataset[train_size:len(dataset),:], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Plot training and testing loss
plt.figure()
plt.figure(figsize=(14,10))
plt.plot(numpy.array(concretemodel.history['loss']), 'b-', label='train')
plt.plot(numpy.array(concretemodel.history['val_loss']), 'm-', label='test')
plt.title('Training and Testing Loss')
plt.legend()
plt.show()

# Save model
model.save("Post-Disc13.h5")
#
# Open old model
#model = tf.keras.models.load_model("model1.h5")
## Save model
#model.save("model2.h5")
#
## Open old model
#model = tf.keras.models.load_model("model1.h5")


##Monte Carlo technique to plot the P-I Diagram
#Drawing P-I Diagram
TR = 1000 #Initial Prediction Trials

WALLNO = 2

RU_Ktemps = datasetX[WALLNO,2:5].reshape(1,1,3) #Select Ru & K of Sample Required to draw P_I diagram for
RU_K_s = numpy.zeros((TR,1,3))
RU_Ktemp = dataframeX[WALLNO,2:5].reshape(1,1,3)
RU_K = numpy.zeros((TR,3))

for t in range(TR): #Filling numpy with same value TR times
    RU_K_s[t] = RU_Ktemps
    RU_K[t] = RU_Ktemp
      

#Filling Pressure With Random Values, and Scaling them
Ptemp = numpy.random.randint(low = 4, high = 285 , size=(TR,1))
P = Ptemp.reshape((TR,1))
P_s = (scalerX1.fit_transform(Ptemp)).reshape(Ptemp.shape[0])
P_s = P_s.reshape(TR,1,1)

#Filling Impulse With Random Values, and Scaling them
Itemp = numpy.random.randint(low = 30, high = 345 , size=(TR,1))
I = Itemp.reshape((TR,1))
I_s = (scalerX2.fit_transform(Itemp)).reshape(Itemp.shape[0])
I_s = I_s.reshape(TR,1,1)

#Invidual Values are ready
P_I_s = numpy.append(P_s,I_s, axis = 1).reshape(TR,1,2)
P_I = numpy.append(P,I, axis = 1)

RKPI_s = numpy.concatenate([P_I_s, RU_K_s], -1)
RKPI = numpy.append(P_I, RU_K , axis=1)


BestModel = tf.keras.models.load_model(
    '/Users/islamtorky/Desktop/Neural Networks/Post-Disc11.h5',
    custom_objects=None,
    compile=True
)


b_s = BestModel.predict(RKPI_s.reshape(TR,1,5))
b = scaler.inverse_transform(b_s.reshape(-1,1))
KABOOM = numpy.append(RKPI, b , axis=1) 

#________________________________________________________________________________________________________________

#Start Loop to find adequate amount of combined P-I Values to plot

LOD = 3 #Damage Criteria in degrees

cc = pd.DataFrame()

for t in range (TR):
    if KABOOM[:,4][t] == (LOD):
        cc[t] = KABOOM[t]

 
cc = cc.T
xx = len(cc.columns)
print(xx)

while xx < TR:
   
    Ptemp = numpy.random.randint(low = 4, high = 285 , size=(TR,1))
    P = Ptemp.reshape((TR,1))
    P_s = (scalerX1.fit_transform(Ptemp)).reshape(Ptemp.shape[0])
    P_s = P_s.reshape(TR,1,1)
    Itemp = numpy.random.randint(low = 30, high = 345 , size=(TR,1))
    I = Itemp.reshape((TR,1))
    I_s = (scalerX2.fit_transform(Itemp)).reshape(Itemp.shape[0])
    I_s = I_s.reshape(TR,1,1)
    P_I_s = numpy.append(P_s,I_s, axis = 1).reshape(TR,1,2)
    P_I = numpy.append(P,I, axis = 1)

    RKPI_s = numpy.concatenate([P_I_s, RU_K_s], -1)
    RKPI = numpy.append(P_I, RU_K , axis=1)

    b_s = BestModel.predict(RKPI_s.reshape(TR,1,5))
    b = scaler.inverse_transform(b_s.reshape(-1,1))
    KABOOM = numpy.append(RKPI, b , axis=1) 
 

    for t in range (TR):
        if KABOOM[:,5][t] >= (LOD - 0.15) and KABOOM[:,5][t] <= (LOD + 0.15):
            cc[t] = KABOOM[t]
    
    xx = len(cc.columns)
    if xx == TR:
        break

cc = cc.T


PP = cc.iloc[:,0]
II = cc.iloc[:,1]

plt.scatter(II,PP)