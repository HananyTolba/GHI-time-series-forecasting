# %%
from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as datetime

# %%

#set current working directory
os.chdir('/Users/hanany/Downloads/Practical-Time-Series-Analysis-master')
#Read the dataset into a pandas.DataFrame
df = pd.read_csv('/Users/hanany/Desktop/lstm-GHI/data/dataGHIDNI_1ans_10m.txt',sep=',', engine='python')



# %%
print('Shape of the dataframe:', df.shape)

# %%
print(df.head())
print(df.info())

# %%
df['DateTime']=pd.to_datetime(df['DateTime'])

print(df.head())
print(df.info())

# %%


# %%


# %%
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(df['GHI'])
g.set_title('Box plot of GHI')

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_GHI'] = scaler.fit_transform(np.array(df['GHI']).reshape(-1, 1))

# %%
# split into train and test sets
train_size = int(len(df.GHI) * 0.67)
df_train, df_val = df[:train_size], df[train_size+1:]



print('Shape of train:', df_train.shape)
print('Shape of test:', df_val.shape)

# %%
df_train.head()

# %%
df_val.head()

# %%
df_val.reset_index(drop=True, inplace=True)

# %%
"""
The train and validation time series of standardized PRES are also plotted.
"""

plt.figure(figsize=(5.5, 5.5))
g = plt.plot(df_train['scaled_GHI'], color='b')
#g.set_title('Time series of scaled Air Pressure in train set')
#g.set_xlabel('Index')
#g.set_ylabel('Scaled Air Pressure readings')
#plt.savefig('plots/ch5/B07887_05_03.png', format='png', dpi=300)

plt.figure(figsize=(5.5, 5.5))
g = plt.plot(df_val['scaled_GHI'], color='r')
#g.set_title('Time series of scaled Air Pressure in validation set')
#g.set_xlabel('Index')
#g.set_ylabel('Scaled Air Pressure readings')
#plt.savefig('plots/ch5/B07887_05_04.png', format='png', dpi=300)

# %%
def makeXy(ts, nb_timesteps):
    """
    Input: 
           ts: original time series
           nb_timesteps: number of time steps in the regressors
    Output: 
           X: 2-D array of regressors
           y: 1-D array of target 
    """
    X = []
    y = []
    for i in range(nb_timesteps, ts.shape[0]):
        X.append(list(ts.loc[i-nb_timesteps:i-1]))
        y.append(ts.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y

# %%
dimInput=10
X_train, y_train = makeXy(df_train['scaled_GHI'], dimInput)
print('Shape of train arrays:', X_train.shape, y_train.shape)

# %%
X_val, y_val = makeXy(df_val['scaled_GHI'], dimInput)
print('Shape of validation arrays:', X_val.shape, y_val.shape)

# %%
from keras.layers import Dense, Input, Dropout
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

# %%
#Define input layer which has shape (None, 7) and of type float32. None indicates the number of instances
input_layer = Input(shape=(dimInput,), dtype='float32')

# %%
#Dense layers are defined with linear activation
dense1 = Dense(64, activation='relu')(input_layer)
dense2 = Dense(32, activation='relu')(dense1)
dense3 = Dense(32, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)

# %%
dropout_layer = Dropout(0.2)(dense4)

# %%
#Finally, the output layer gives prediction for the next day's air pressure.
output_layer = Dense(1, activation='relu')(dropout_layer)

# %%
ts_model = Model(inputs=input_layer, outputs=output_layer)
ts_model.compile(loss='mean_squared_error', optimizer='adam')
ts_model.summary()

# %%
save_weights_at = os.path.join('/Users/hanany/Downloads/Practical-Time-Series-Analysis-master', 'GHI_MLP_weights.{epoch:02d}-{val_loss:.4f}.hdf5')
print(save_weights_at)

# %%
save_best = ModelCheckpoint(save_weights_at, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False, mode='min',
                            period=1)


# %%
ts_model.fit(x=X_train, y=y_train, batch_size=500, epochs=200,
             verbose=1, callbacks=[save_best], validation_data=(X_val, y_val),
             shuffle=True)

# %%
best_model = load_model(os.path.join('/Users/hanany/Downloads/Practical-Time-Series-Analysis-master',
                                     'GHI_MLP_weights.20-0.0015.hdf5'))
preds = best_model.predict(X_val)
pred_PRES = scaler.inverse_transform(preds)
pred_PRES = np.squeeze(pred_PRES)

# %%


# %%
#Let's plot the first 50 actual and predicted values of air pressure.
tailPlot=150
plt.figure(figsize=(5.5, 5.5))
plt.plot(range(tailPlot), df_val['GHI'].loc[dimInput:tailPlot+dimInput-1], linestyle='-', marker='*', color='b')
plt.plot(range(tailPlot), pred_PRES[:tailPlot], linestyle='-', marker='.', color='r')
plt.legend(['Actual','Predicted'], loc=2)
plt.title('Actual vs Predicted Air Pressure')
plt.ylabel('GHI')

# %%
from sklearn.metrics import r2_score,mean_squared_error
r2 = r2_score(df_val['GHI'].loc[dimInput:], pred_PRES)
print('R-squared for the validation set:', round(r2,4))

# calculate root mean squared error
testScore = np.sqrt(mean_squared_error(df_val['GHI'].loc[dimInput:], pred_PRES))/np.mean(df_val['GHI'].loc[dimInput:])
print('Test Score: %.2f nRMSE' % (testScore))


#trainScore = np.sqrt(mean_squared_error(df_train['GHI'].loc[:dimInput], pred_PRES))/np.mean(df_val['GHI'].loc[dimInput:])
print('Test Score: %.2f RMSE' % (np.sqrt(mean_squared_error(df_val['GHI'].loc[dimInput:], pred_PRES))))
#testScore = np.sqrt(mean_squared_error(testYI[0], testPredictI[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))
#print(len(df_val['GHI'].loc[dimInput:]),len(pred_PRES))


# %%
#import plotly
#import plotly.graph_objs as go

#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#init_notebook_mode(connected=True)
#iplot([{"x": time, "y": df_val['GHI'].loc[dimInput:]},{"x": time, "y": pred_PRES}])
#iplot([{"x": time, "y": ssn}])

# %%


# %%
