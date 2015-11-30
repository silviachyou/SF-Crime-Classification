
# coding: utf-8

# In[2]:

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np
 
#Load Data with pandas, and parse the first column into datetime
train=pd.read_csv('train2.csv', sep='^', parse_dates = ['Dates'])

print "Data Read. Preprocesing...\n"

# In[29]:

train = train[train.X != -120.5]


# In[30]:

max_Y = max(train.Y)
min_Y = min(train.Y)
max_X = max(train.X)
min_X = min(train.X)


# In[5]:

#train = train.join(((train.X - min_X)/(max_X - min_X) * 8), rsuffix='_grid')


# In[31]:

train['grid_X'] = pd.DataFrame((train.X - min_X)/(max_X - min_X) * 8).astype(int)


# In[32]:

train['grid_Y'] = ((train.Y - min_Y)/(max_Y - min_Y) * 8).astype(int)


# In[33]:

grid_features = pd.get_dummies(train.grid_X).join(pd.get_dummies(train.grid_Y),lsuffix="_X",rsuffix="_Y")


# In[34]:



#Convert crime labels to numbers
le_crime = preprocessing.LabelEncoder()
crime = le_crime.fit_transform(train.Category)
 
#Get binarized weekdays, districts, and hours.
days = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
hour = train.Dates.dt.hour
hour = pd.get_dummies(hour,prefix="hour") 
month = train.Dates.dt.month
month = pd.get_dummies(month,prefix="month")
 
#Build new array
train_data = pd.concat([hour, days, month, grid_features, district], axis=1)
train_data['crime']=crime


# In[36]:

features = ['hour_0','hour_1','hour_2','hour_3','hour_4','hour_5','hour_6','hour_7','hour_8','hour_9',
            'hour_10','hour_11','hour_12','hour_13','hour_14','hour_15','hour_16','hour_17','hour_18','hour_19',
            'hour_20','hour_21','hour_22','hour_23','month_1','month_2','month_3','month_4','month_5','month_6',
            'month_7','month_8','month_9','month_10','month_11','month_12',
            '0_X','1_X','2_X','3_X','4_X','5_X','6_X','7_X','0_Y','1_Y','2_Y','3_Y','4_Y','5_Y','6_Y','7_Y',
            'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
 'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
 


print "Running Random Forest Classifier...\n"


# In[37]:

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, max_depth=None)
model.fit(train_data[features], train_data['crime'])
#predicted = np.array(model.predict_proba(valid_data[features]))
#print log_loss(valid_data['crime'], predicted) 


print "Preprocessing Test Data...\n"

# Change to the name of the test file you are using:

test=pd.read_csv('kaggle_test_1.csv', parse_dates = ['Dates'])

test.loc[test.X == -120.5, 'X'] = max_X
test.loc[test.Y == 90,'Y'] = max_Y

#test_crime = le_crime.fit_transform(test.Category)

#Repeat for test data
days = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)

hour = test.Dates.dt.hour
hour = pd.get_dummies(hour,prefix="hour") 
month = test.Dates.dt.month
month = pd.get_dummies(month,prefix="month")

test['grid_X'] = ((test.X - min_X)/(max_X - min_X) * 8).astype(int)
test['grid_Y'] = ((test.Y - min_Y)/(max_Y - min_Y) * 8).astype(int)
grid_features = pd.get_dummies(test.grid_X).join(pd.get_dummies(test.grid_Y),lsuffix="_X",rsuffix="_Y")

test_data = pd.concat([hour, days, month, grid_features, district], axis=1)
#test_data['crime']=test_crime

print "Running model on Test data... \n"

predicted = model.predict_proba(test_data[features])
 

print "Writing results to file... \n"
#Write results
result=pd.DataFrame(predicted, columns=le_crime.classes_)
result.to_csv('testResult.csv', index = True, index_label = 'Id' )





