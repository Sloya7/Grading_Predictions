#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sklearn
import os.path
from os import path


# In[ ]:


#reads in the training data and data of cards that need a grade assigned
train_data = pd.read_csv('card_training_data.csv')


user_data = input("Enter File Path for csv containing card data OR type 'skip' to use the file named self_grade_sheet")

  
if user_data == 'skip':
    predict_data  = pd.read_csv('self_grade_sheet.csv')

else:
    if path.exists(user_data) != True:
        print("File not found. Make sure file is in same directory or full file path is used.")
        user_data = input("Enter File Path for csv containing card data:")

    if path.exists(user_data) != True:
        print("File path still not found. Ending Program")
        exit()

    





if len(user_data) > 1 and user_data != 'skip':
    predict_data = pd.read_csv(user_data)
else:
    predict_data = pd.read_csv('self_grade_sheet.csv')




# In[ ]:


print('Number of NaNs filled:', predict_data.isna().sum().sum())
predict_data = predict_data.fillna(0)


# In[ ]:


# providing guidence on what percents allow for a X maximum grade. 
# 10 = 55/45,  9 = 60/40, 8 = 65/35, 7 = 70/30, 6 = 75/25, 4.5 = 85/15, 3.5 = 90/10

#takes in a column of percents and assigns a grade
def percent_to_num(df_col):
    for r in df_col.index:
        num = df_col[r]
        if type(num) == str:
            try:
                num.astype(float)
            except:
                print('Non Number Values present. Check self graded data sheet')
                print('Problem with Value:',num)
                print('Row:', r, 'Column:', df_col.name)
        if num >= .45 and num <= .55:
            df_col[r] = 10
        elif num >= .4 and num <= .6:
            df_col[r] = 9
        elif num >= .35 and num <= .65:
            df_col[r] = 8
        elif num >= .30 and num <= .7:
            df_col[r] = 7
        elif num >= .25 and num <= .75:
            df_col[r] = 6
        elif num >= .15 and num <= .85:
            df_col[r] = 4.5
        elif num >= .1 and num <= .9:
            df_col[r] = 3.5
        else:
            df_col[r] = 1


# In[ ]:


#list of columns that need the function performed on. Should be 4 columns that contain the percents
list_to_change = [predict_data.tb_split_percent, predict_data.side_split_percent, train_data.tb_split_percent, train_data.side_split_percent]

#executes the function on the above list of columns
for df in range(len(list_to_change)):
    percent_to_num(list_to_change[df])


# In[ ]:


# run to check for successful conversion
#predict_data.tb_split_percent


# In[ ]:


# creates test data and removes the rows from the training data
test_slice = round(len(train_data)*.2)
test_rows = np.random.choice(train_data.ID, test_slice)


test_data = train_data[train_data['ID'].isin(test_rows)]
train_data = train_data[~train_data['ID'].isin(test_rows)]


# In[ ]:


#removes columns that are for labeling like 'Set', 'Card Number' etc.
train_columns = train_data.columns[11:]
train_values_data = train_data[train_columns]
test_values_data = test_data[train_columns]


# In[ ]:


#establish data and desired trait to train for
x_train = train_values_data
x_test = test_values_data
y_train = train_data['Grade']
y_test = test_data['Grade']


# In[ ]:


# creates dataframe with the same dimensions
pred_columns = predict_data.columns[11:]
pred_data = predict_data[pred_columns]


# In[ ]:


#create df that houses a bootstraped training values
boot_df = pd.DataFrame()


# In[ ]:


# import modeler
from sklearn.ensemble import RandomForestRegressor


# ### Grid Search and Results
# running a grid search, RandomForestRegressor on 7/8 was the best model with 
# Best parameters: {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 10}
# #Best score: 0.6459180794774193
# kept in script for future assessment
# 
# from sklearn.model_selection import GridSearchCV, train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=5)
# 
# param_grid = {
#     'n_estimators': [10, 50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# 
# grid_model = RandomForestRegressor(random_state = 5)
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')
# grid_search.fit(x_train, y_train)
# 
# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Best score: {grid_search.best_score_}")
# 

# In[ ]:


# uses latest model suggestions to run data

model = RandomForestRegressor(n_estimators = 10, max_depth = 10, min_samples_leaf = 2, min_samples_split=2)
trained_model = model.fit(x_train, y_train)
predict_data['Grade'] = trained_model.predict(pred_data)


# In[ ]:


# for the predicted grades, if the 
for g in range(len(predict_data)):
    if predict_data['Grade'][g] > predict_data['tb_split_percent'][g] or predict_data['Grade'][g] > predict_data['side_split_percent'][g]:
        if predict_data['tb_split_percent'][g] > predict_data['side_split_percent'][g]:
            predict_data['Grade'][g] = predict_data['side_split_percent'][g]
        else:
            predict_data['Grade'][g] = predict_data['tb_split_percent'][g]

# round numbers for easier viewing
predict_data.loc[:,'Grade'] = predict_data['Grade'].round(1)


# In[ ]:


#save data to CSV
predict_data.to_csv('Predicted_Grades.csv')


# In[ ]:




