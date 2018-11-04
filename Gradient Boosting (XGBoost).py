
# coding: utf-8

# # Gradient Boosting (XGBoost) and Hyperparameter tuning using Python
# 
# ##### This project is to demonstrate how to do cross validation to prevent over-fitting and then use Hyperparameter tuning to improve the prediction accuracy further
# 
# ##### We will try and predict Customer Churn using XGBoost

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Read in the data
df = pd.read_csv(r'C:\Users\amitr\OneDrive\Desktop\Deep Learning\Data\Churn_Modelling.csv')
df.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True)
df.head()


# In[3]:


df.info()


# In[3]:


df.describe()


# ### Basic data preprocessing 

# In[4]:


df['Balance'].plot(kind='hist')
plt.show()


# In[5]:


# Let's create a new variable to capture if a person has any balance
df['Has_Balance'] = np.where(df['Balance'] > 0, 1, 0)
df['Has_Balance'].value_counts(normalize=True) * 100


# In[6]:


# Let's scale the relevant columns
from sklearn.preprocessing import StandardScaler

scaled_cols = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
sc = StandardScaler()

df_scaled_cols = sc.fit_transform(df[scaled_cols])

df_final = df.copy()
df_final[scaled_cols] = df_scaled_cols
df_final.head()


# In[7]:


# One hot encoding
df_final = pd.get_dummies(df_final, drop_first=True)
df_final.head()


# In[8]:


# Split into train and test sets

from sklearn.model_selection import train_test_split

y = df_final['Exited']
X = df_final.drop('Exited', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)


# ### Create a basic initial model using XGBoost

# In[19]:


# Create a Decision Tree model

from xgboost import XGBClassifier

mod = XGBClassifier(random_state=0)
mod.fit(X_train, y_train)


# In[20]:


# Predict the results

from sklearn.metrics import confusion_matrix, accuracy_score

pred_train = mod.predict(X_train)
pred_test = mod.predict(X_test)

# Check the accuracy
print("Training accuracy : ", accuracy_score(y_train, pred_train))
print("Testing accuracy : ", accuracy_score(y_test, pred_test))


# In[17]:


# Let's prevent overfitting by using cross-validation
from sklearn.cross_validation import cross_val_score, KFold

k_fold = KFold(len(y_train), n_folds=10, random_state=0, shuffle=True)

avg = cross_val_score(mod, X_train, y_train, cv=k_fold, n_jobs=1)
print ('The accuracies on each fold are : ', avg)
print ("\nAverage accuracy on training set : ", avg.mean())


# ## Hyperparameter tuning
# 
# ##### Let's do some hyperparameter tuning to see if we can improve it further

# In[14]:


# Using GridSearch for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Set the hyperparameter values
parameters = [{'learning_rate':[0.01,0.05,0.1,0.2, 0.3, 0.5, 1.0],
              'max_depth':[3,4,5,6,10],
              'min_child_weight':[1,2,3]}]
    
mod = XGBClassifier()

grid_search = GridSearchCV(estimator=mod,
                           param_grid=parameters, 
                           scoring='accuracy',
                           n_jobs=-1)

# Fit the training data for the models
grid_search = grid_search.fit(X_train, y_train)


# In[15]:


# Display the best score and the optimal parameters

print ("Best accuracy : ", grid_search.best_score_)
print ("\n\nBest parameters : ", grid_search.best_params_)


# In[16]:


# Display the results on the test set

pred_y = grid_search.predict(X_test)

print ('Confusion Matrix : \n', confusion_matrix(y_test,pred_y))
print ('\n Test set accuracy : ', accuracy_score(y_test, pred_y))


# ##### There is not much difference in the accuracy on the test set.That is because XGBoost by default choose excelent parameters and so hyperparameter tuning did little to improve it further. Anyways, it's an excellent machine learning algorithm.
