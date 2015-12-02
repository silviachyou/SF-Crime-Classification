
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

grid_size = 20

train['grid_X'] = pd.DataFrame((train.X - min_X)/(max_X - min_X) * grid_size).astype(int)


# In[32]:

train['grid_Y'] = ((train.Y - min_Y)/(max_Y - min_Y) * grid_size).astype(int)


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
            '0_X','1_X','2_X','3_X','4_X','5_X','6_X','7_X','8_X','9_X','10_X','11_X',
            '12_X','13_X','14_X','15_X','16_X','17_X','18_X','19_X',
            '0_Y','1_Y','2_Y','3_Y','4_Y','5_Y','6_Y','7_Y','8_Y','9_Y','10_Y','11_Y',
            '12_Y','13_Y','14_Y','15_Y','16_Y','17_Y','18_Y','19_Y',
            'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
 'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
 


print "Running Random Forest Classifier...\n"


# In[37]:

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, max_depth=20)
model.fit(train_data[features], train_data['crime'])
#predicted = np.array(model.predict_proba(valid_data[features]))
#print log_loss(valid_data['crime'], predicted) 


print "Preprocessing Test Data...\n"

# Change to the name of the test file you are using:

test=pd.read_csv('test.csv', parse_dates = ['Dates'])

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

test['grid_X'] = ((test.X - min_X)/(max_X - min_X) * grid_size).astype(int)
test['grid_Y'] = ((test.Y - min_Y)/(max_Y - min_Y) * grid_size).astype(int)
grid_features = pd.get_dummies(test.grid_X).join(pd.get_dummies(test.grid_Y),lsuffix="_X",rsuffix="_Y")

test_data = pd.concat([hour, days, month, grid_features, district], axis=1)
#test_data['crime']=test_crime

print "Running model on Test data... \n"

predicted = model.predict_proba(test_data[features])
 

print "Writing results to file... \n"
#Write results
result=pd.DataFrame(predicted, columns=le_crime.classes_)
result.to_csv('testResult.csv', index = True, index_label = 'Id' )



