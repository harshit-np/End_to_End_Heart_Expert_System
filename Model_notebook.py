#!/usr/bin/env python
# coding: utf-8

# ## HEART DISEASE PREDICTION 

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import joblib
from sklearn import metrics


# In[16]:


df=pd.read_csv("C:/Users/user/Downloads/heart_disease_uci.csv")


# In[17]:


df.head()


# In[18]:


df.shape


# In[19]:


df.describe()


# In[20]:


df.shape


# ## Checking for any null values in the dataset

# So here we wil handle all the missing values.If we found any missing record in the patient record we will remove the entire record of the patient.Also shuffling the data is also a good practice.

# In[21]:


df.isna().sum()*100/df.shape[0]


# In[23]:


df.dropna(inplace= True )
df=df.sample(frac=1)


# In[24]:


df.shape


# In[25]:


plt.boxplot(x=df.age)


# In[26]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, fmt='.1f')
plt.show()


# ## PREPROCESSING

# In[27]:


df.dtypes


# In[28]:


df.columns


# In[29]:


df=df.drop(['dataset'],axis=1)


# In[30]:


df.sex.value_counts()


# In[31]:


df.fbs.value_counts()


# In[32]:


df.num.value_counts()


# In[33]:


df.ca.value_counts()


# In[34]:


df.exang.value_counts()


# Here the Target feature i.e. num is categorise in 5 categories .But for identifying simply the presence of disease, we will take binary classification. We will convert the feature num in 0/1

# In[35]:


df['target']=(df['num']>0)*1
df['exang']=(df['exang'])*1
df['fbs']=(df['fbs'])*1
df['sex']=(df['sex']== 'Male')*1


# In[36]:


df.drop(['num','id'],axis=1, inplace=True)


# In[37]:


df.head(40)


# In[38]:


df.cp.value_counts()


# In[39]:


df.columns


# In[40]:


df1=pd.get_dummies(df.drop(['target','fbs','exang'],axis=1),drop_first=False)
df1['fbs'] = df['fbs'].copy()
df1['exang'] = df['exang'].copy()
df1.columns


# In[41]:


df1.columns


# In[42]:


df1.head(30)


# ## LOGESTIC REGRESSION

# In[43]:


#splitiing the data in train and test data
from sklearn.model_selection import train_test_split
X=df1
Y=df['target']
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=.2, random_state=1)


# In[79]:


#nomralization of the dataframe using the min-max method
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
X_test_un = scaler.inverse_transform(X_test)
# X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
# X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values
joblib.dump(scaler,"C:/Users/user/Downloads/scaler_hd.pkl")


# ## Model Fitting

# In[70]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred=logreg.predict(X_test)
y_p = logreg.predict(X_train)
print('Test Accuracy {:.2f}%'.format(metrics.precision_score(Y_test,Y_pred)*100))
print("Train Accuracy:{:.2f}% ".format(metrics.precision_score(Y_train, y_p)*100))


# In[ ]:





# In[71]:


joblib.dump(logreg,"C:/Users/user/Downloads/logreg_hd.pkl")


# In[72]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
cnf_matrix


# In[73]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks)
plt.yticks(tick_marks)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), cmap="YlGnBu")
ax.xaxis.set_label_position("top")
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[74]:


print("Train Accuracy:{:.2f}% ".format(metrics.accuracy_score(Y_train, y_p)*100))
print("Test Accuracy:{:.2f}% ".format(metrics.accuracy_score(Y_test, Y_pred)*100))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[75]:


df.head()


# In[76]:


Y_pred_proba = logreg.predict_proba(X_train)[:,1]
fpr, tpr, _ = metrics.roc_curve(Y_train,  Y_pred_proba)
auc = metrics.roc_auc_score(Y_train, Y_pred_proba)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc=4)
plt.show()
Y_pred_proba


# In[77]:


Y_pred_proba = logreg.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(Y_test,  Y_pred_proba)
auc = metrics.roc_auc_score(Y_test, Y_pred_proba)
plt.plot(fpr,tpr,label="auc="+str(auc))
plt.legend(loc=4)
plt.show()
Y_pred_proba


# In[80]:


import numpy as np
with np.printoptions(threshold=np.inf):
    print(X_test_un)


# In[66]:


df1.head()


# In[ ]:


0.3125     1.         0.26530612 0.21766562 0.55725191 0.4516129
  0.         1.         0.         0.         0.         0.
  1.         0.         1.         0.         0.         1.
  0.         0.         0.         1.        


# In[81]:


df1.columns


# In[57]:


from sklearn.svm import SVC
svm = SVC(kernel='linear', random_state = 112)
svm.fit(X_train, Y_train)
print("Test SVC accuracy: {:.2f}%".format(svm.score(X_test, Y_test)*100))


# ## Validation Prediction 

# In[152]:


import json
def preprocessing(df_tmp,feat_cols):
    df_tmp['sex'] = (df_tmp['sex']== 'Male')*1
    df_tmp['exang']= df_tmp['exang'].apply(lambda x:1 if x=='True' else 0)
    df_tmp['fbs']= df_tmp['fbs'].apply(lambda x:1 if x=='True' else 0)
    
    for col in ['cp','restecg','slope','thal']: 
        col_val = df_tmp[col].iloc[0]
        df_tmp = df_tmp.rename(columns ={col:col_val})
        df_tmp[col_val] = 1

    for col in feat_cols:
        if col not in df_tmp.columns:
            df_tmp[col] = 0
            
    scaler = joblib.load('C:/Users/user/Downloads/scaler_hd.pkl')
    df_tmp_std = scaler.transform(df_tmp[feat_cols])
    
    return df_tmp_std

with open("C:/Users/user/Downloads/features.json") as file:
    valid_data = json.load(file)
    
#valid_data = {'patientName':'Harshit Bansal', 'age':'20','sex':'Male','cp':'cp_asymptomatic','trestbps':'135','fbs':'True',
#         'chol':'34','restecg':'restecg_normal','thalach':'97','exang':'True','oldpeak':'23','slope':'slope_downsloping'
#         ,'ca':'2','thal':'thal_reversable defect','submit':''}

features = ['age', 'sex', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca',
           'cp_asymptomatic', 'cp_atypical angina', 'cp_non-anginal',
           'cp_typical angina', 'fbs', 'restecg_lv hypertrophy',
           'restecg_normal', 'restecg_st-t abnormality',  'exang',
           'slope_downsloping', 'slope_flat', 'slope_upsloping',
           'thal_fixed defect', 'thal_normal', 'thal_reversable defect']

valid_df = preprocessing(pd.DataFrame(valid_data,index=[0]),features)
print(valid_df)
model = joblib.load("C:/Users/user/Downloads/logreg_hd.pkl")
print(f"Heart Disease Prediction : {model.predict(valid_df)[0]}")
print(f"Heart Disease Prediction Probability : {int(model.predict_proba(valid_df)[0][1]*100)}%")


# In[136]:


valid_data = {'patientName':'Harshit Bansal', 'age':'20','sex':'Male','cp':'cp_asymptomatic','trestbps':'135','fbs':'True',
             'chol':'34','restecg':'restecg_normal','thalach':'97','exang':'True','oldpeak':'23','slope':'slope_downsloping'
             ,'ca':'2','thal':'thal_reversable defect','submit':''}
json.dumps(valid_data)


# In[ ]:





# In[155]:





# In[ ]:





# In[ ]:




