

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:14:35 20222

@author: haselebe
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import  accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import codecs, json
import tempfile
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



"""
Collect and prepare the dataset and Reading all the seasons(10)
 data into the Dataframe
"""
df1=pd.read_csv(r"C:\Users\44776\Downloads\season-0910_csv.csv")
df2=pd.read_csv(r"C:\Users\44776\Downloads\season-1011_csv.csv")
df3=pd.read_csv(r"C:\Users\44776\Downloads\season-1112_csv.csv")
df4=pd.read_csv(r"C:\Users\44776\Downloads\season-1213_csv.csv")
df5=pd.read_csv(r"C:\Users\44776\Downloads\season-1415_csv.csv")
df6=pd.read_csv(r"C:\Users\44776\Downloads\season-1516_csv.csv")
df7=pd.read_csv(r"C:\Users\44776\Downloads\season-1617_csv.csv")
df8=pd.read_csv(r"C:\Users\44776\Downloads\season-1718_csv.csv")
df9=pd.read_csv(r"C:\Users\44776\Downloads\season-1819_csv.csv")
df10=pd.read_csv(r"C:\Users\44776\Downloads\season-1314_csv.csv")

df1["season"]="0910"
df2["season"]="1011"
df3["season"]="1112"
df4["season"]="1213"
df5["season"]="1415"
df6["season"]="1516"
df7["season"]="1617"
df8["season"]="1718"
df9["season"]="1819"
df10["season"]="1314"

df1.to_csv("df1.csv")
df2.to_csv("df2.csv")
df3.to_csv("df3.csv")
df4.to_csv("df4.csv")
df5.to_csv("df5.csv")
df6.to_csv("df6.csv")
df7.to_csv("df7.csv")
df8.to_csv("df8.csv")
df9.to_csv("df9.csv")
df10.to_csv("df10.csv")

merge_df = pd.concat(
  map(pd.read_csv, ['df1.csv', 'df2.csv','df3.csv','df4.csv',
                    'df5.csv','df6.csv','df7.csv','df8.csv',
                    'df9.csv','df10.csv']), ignore_index=True)


"""Create Home Win, Draw Win and Away Win columns"""

merge_df= merge_df.assign(homeWin=lambda merge_df: merge_df.apply(lambda row: 1 if row.FTHG > row.FTAG else 0, axis='columns'),
              draw=lambda merge_df: merge_df.apply(lambda row: 1 if row.FTHG == row.FTAG else 0, axis='columns'),
              awayWin=lambda merge_df: merge_df.apply(lambda row: 1 if row.FTHG < row.FTAG else 0, axis='columns'))

print(merge_df )

merge_df=merge_df.fillna(0)

merge_df.to_csv("Artificial data.csv")


merge_df.shape
merge_df["new_FTR"]=merge_df["FTR"].astype("category").cat.codes

merge_df["new_HTR"]=merge_df["HTR"].astype("category").cat.codes
encoder = LabelEncoder()
merge_df["FTHG"] = encoder.fit_transform(merge_df["FTHG"])

encoder = LabelEncoder()
merge_df["FTHG"] = encoder.fit_transform(merge_df["FTHG"])

merge_df["Target"]=(merge_df["FTR"]=="A").astype("int")

merge_df["Target_HTR"]=(merge_df["HTR"]=="A").astype("int")
merge_df=merge_df.fillna(0)


data=[6, 8, 9,12,13, 14,15, 16, 17, 18,19,20, 21,72,79,80,81,82,83,84,85]
x=merge_df.iloc[:,data]

fig, chart = plt.subplots()
data = merge_df['FTR'].value_counts()
points = data.index
frequency = data.values
chart.bar(points, frequency)
chart.set_title('Frequency of different results in the English Premiership')
chart.set_xlabel('Result Type')
chart.set_ylabel('Frequency')

epl_df_objects=merge_df.copy()
merge_df["Date"] = pd.to_datetime(merge_df["Date"], infer_datetime_format=True)
merge_df['matchDay'] = merge_df['Date'].dt.day_name()

epl_df_objects.head()

dd=[2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21, 
    79,80,81,82,83,84,85,86]
data1=merge_df.iloc[:,dd]

x1=data1.copy()
x1 = pd.get_dummies(x1, columns=['HomeTeam'], prefix = ['HomeTeam'])
x1 = pd.get_dummies(x1, columns=['AwayTeam'], prefix = ['AwayTeam'])
x1 = pd.get_dummies(x1, columns=['HTR'], prefix = ['HTR'])
x1 = pd.get_dummies(x1, columns=['matchDay'], prefix = ['matchDay'])
x1.head()


label_encoder = LabelEncoder()
x1['FTR']= label_encoder.fit_transform(x1['FTR'])
print('Unique values for our label are: ', x1['FTR'].unique())
print('if the home team wins the label is ',x1['FTR'][0])
print('if the away team wins the label is ', x1['FTR'][2])
print('if there is a tie the label is ', x1['FTR'][3])
label = x1['FTR']
print('the result for the match in row 149 is ', label[102])
print(x1.iloc[:,3:111])
features = x1.iloc[:,3:111]

"""Separate the data for training and test
"""

y=np.ravel(label)
X = features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                    shuffle=False)
print("The shape of X_train is " + str(X_train.shape))
print("The size of y_train is " + str(y_train.shape))
print("The size of X_test set is " + str(X_test.shape))
print("The size of y_test is " + str(y_test.shape))


import base64

# one hot-encoding y_train and y_test
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)
print("The size of y_train is " + str(y_train.shape))
print("The size of y_test is " + str(y_test.shape))
print(y_train[0])

model = tf.keras.models.Sequential([
tf.keras.layers.Dense(330, input_dim=102, activation='relu'),
tf.keras.layers.Dense(10, input_dim=330, activation='relu'),
tf.keras.layers.Dense(3,activation='softmax')])
model.summary()
model.compile(loss = 'categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

X_train = np.expand_dims(X_train, axis=-1)


history = model.fit(X_train, y_train, epochs=100)

#Accuracy history
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


#Loss history
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


score = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

Xnew = np.array([[ 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
print(Xnew.shape)
# make a prediction
ynew = np.argmax(model.predict(Xnew), axis=-1)
# show the inputs and predicted outputs
print("X = %s " % Xnew)
print("Prediction = %s" % ynew[0])

big_clubs = ['Liverpool', 'Man City', 'Man United', 'Chelsea', 'Arsenal']
home_win_rates_5 = merge_df[merge_df.HomeTeam.isin(big_clubs)].groupby(['HomeTeam',
                                                    'season']).homeWin.mean()
away_win_rates_5 =merge_df[merge_df.AwayTeam.isin(big_clubs)].groupby(['AwayTeam', 
                                                    'season']).awayWin.mean()

hga_top_5 = home_win_rates_5 - away_win_rates_5

hga_top_5.unstack(level=0)

import seaborn as sns
sns.lineplot(x='season', y='HGA', hue='team', 
   data=hga_top_5.reset_index().rename(columns={0: 'HGA', 'HomeTeam': 'team'}))
plt.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.45, -0.2))
plt.title("Among the top 5 clubs", fontsize=14)
plt.show()


win_rates = \
(merge_df.groupby('season')
    .mean()
    .loc[:, ['homeWin', 'draw', 'awayWin']])

win_rates


# Set the style
plt.style.use('ggplot')

fig = plt.figure()
ax = fig.add_subplot(111)

home_line = ax.plot(win_rates.homeWin, label='Home Win')
away_line = ax.plot(win_rates.awayWin, label='Away Win')
draw_line = ax.plot(win_rates.draw, label='Draw')
ax.set_xlabel("season")
ax.set_ylabel("Win Rate")
plt.title("Win Rates", fontsize=16)

# Add the legend locations
home_legend = plt.legend(handles=home_line, loc='upper right',
                         bbox_to_anchor=(1, 1))
ax = plt.gca().add_artist(home_legend)
away_legend = plt.legend(handles=away_line, loc='center right',
                         bbox_to_anchor=(0.95, 0.4))
ax = plt.gca().add_artist(away_legend)
draw_legend = plt.legend(handles=draw_line, loc='center right',
                         bbox_to_anchor=(0.95, 0.06))