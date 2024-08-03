#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as pl


# In[2]:


df = pd.read_csv('C:/Users/DAVID/Downloads/Train.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df2 = df.drop(['user_id', 'MRG'],axis='columns')
df2.head()


# In[5]:


cols = ['REGION', "TENURE", 'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 
        'ON_NET', 'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 'REGULARITY', 'TOP_PACK', 'FREQ_TOP_PACK', 'CHURN']

for i in cols:
    print(i)
    print(df2[i].unique())
    print(df2[i].nunique())
    print()


# In[6]:


df2.isnull().sum()


# In[7]:


df2[['ORANGE', 'TIGO', 'ZONE1', 'ZONE2']] = df2[['ORANGE', 'TIGO', 'ZONE1', 'ZONE2']].fillna(0)


# In[8]:


df2.isnull().sum()


# In[9]:


df2['TOP_PACK'] = df2['TOP_PACK'].fillna('No Pack')
df2['FREQ_TOP_PACK'] = df2['FREQ_TOP_PACK'].fillna(0)


# In[10]:


df2.isnull().sum()


# In[11]:


average_montant = df2['MONTANT'].mean()


# In[12]:


df2['MONTANT'] = df2['MONTANT'].fillna(average_montant)
df2.isnull().sum()           


# In[13]:


df2.isnull().sum()   


# 

# In[14]:


df2.head()


# In[15]:


unique_regions = df2['REGION'].dropna().unique()

# Replace NaN values with random choices from the unique regions
df2['REGION'] = df2['REGION'].apply(lambda x: np.random.choice(unique_regions) if pd.isnull(x) else x)


# In[16]:


df2.isnull().sum()   


# In[17]:


df2.head()


# In[18]:


average_revenue = df2['REVENUE'].mean()
# Replace NaN values in the REVENUE column with the average REVENUE
df2['REVENUE'] = df2['REVENUE'].fillna(average_revenue)


# In[19]:


df2.isnull().sum()   


# In[20]:


average_arpu_segment = df2['ARPU_SEGMENT'].mean()
# Replace NaN values in the ARPU_SEGMENT column with the average ARPU_SEGMENT
df2['ARPU_SEGMENT'] = df2['ARPU_SEGMENT'].fillna(average_arpu_segment)


# In[21]:


df2.isnull().sum()   


# In[22]:


average_frequence_rech = df2['FREQUENCE_RECH'].mean()
# Replace NaN values in the FREQUENCE_RECH column with the average FREQUENCE_RECH
df2['FREQUENCE_RECH'] = df2['FREQUENCE_RECH'].fillna(average_frequence_rech)


# In[23]:


df2.isnull().sum()   


# In[24]:


average_frequence = df2['FREQUENCE'].mean()
# Replace NaN values in the FREQUENCE column with the average FREQUENCE
df2['FREQUENCE'] = df2['FREQUENCE'].fillna(average_frequence)


# In[25]:


df2.isnull().sum()   


# In[26]:


average_data_volume = df2['DATA_VOLUME'].mean()
# Replace NaN values in the DATA_VOLUME column with the average DATA_VOLUME
df2['DATA_VOLUME'] = df2['DATA_VOLUME'].fillna(average_data_volume)


# In[27]:


df2.isnull().sum() 


# In[28]:


average_on_net = df2['ON_NET'].mean()
# Replace NaN values in the ON_NET column with the average ON_NET
df2['ON_NET'] = df2['ON_NET'].fillna(average_on_net)


# In[29]:


df2.isnull().sum() 


# In[30]:


df2.head()


# In[31]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df2['TENURE'] = label_encoder.fit_transform(df2['TENURE'])


# In[32]:


df2.head()


# In[33]:


df2.info()


# In[34]:


average_regularity_per_region = df2.groupby('REGION')['REGULARITY'].mean()
sorted_average_regularity = average_regularity_per_region.sort_values(ascending=False)
print(sorted_average_regularity)


# In[35]:


df2['REGION'] = df2['REGION'].map(average_regularity_per_region)
df2.head()


# In[36]:


top_pack_freq = df2.groupby('TOP_PACK')['FREQ_TOP_PACK'].mean()
sorted_top_pack_freq = top_pack_freq.sort_values(ascending=False)
print(sorted_top_pack_freq)


# In[37]:


df2['TOP_PACK'] = df2['TOP_PACK'].map(top_pack_freq)
df2.head()


# In[38]:


df2.info()


# In[39]:


df2['CHURN'] = df2['CHURN'].apply(float)


# In[40]:


final_data = df2
final_data.info()


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


X= final_data.drop(['CHURN'],axis=1)
y = final_data['CHURN']


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 40)


# In[44]:


from sklearn.preprocessing import MinMaxScaler


# In[45]:


mms = MinMaxScaler(feature_range=(0,1))


# In[46]:


# Xtrain = mms.fit_transform(X_train)
# Xtest = mms.fit_transform(X_test)
# Xtrain = pd.DataFrame(X_train)
# Xtest = pd.DataFrame(X_test)
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)


# In[47]:


Results = {'Model':[],'Accuracy':[],'Recall':[],'Precision':[],'F1':[]}


# In[48]:


Results = pd.DataFrame(Results)
Results.head()


# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC


# In[50]:


lr = LogisticRegression()
dc = DecisionTreeClassifier()
rf = RandomForestClassifier()
gbc = GradientBoostingClassifier()
knn = KNeighborsClassifier(n_neighbors = 10)
# svm = SVC()


# In[51]:


Results = pd.DataFrame(columns=['Model','Accuracy','Recall','Precision','F1'])
# model = [lr, dc, rf, gbc, knn, svm]
model = [lr, dc, rf, gbc, knn]
for models in model:
    models.fit(X_train, y_train)
    ypred = models.predict(X_test)
    
    print('Model: ', models)
    print("------------------------------------------------------------------------------------------")
    
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
    
    
    print('Confusion matrix: ', confusion_matrix(y_test,ypred))
    print('Classification report: ', classification_report(y_test,ypred))
    print('accuracy: ', round(accuracy_score(y_test,ypred),2))
    print('precision: ', round(precision_score(y_test,ypred),2))
    print('recall: ', round(recall_score(y_test,ypred),2))
    print('f1: ', round(f1_score(y_test,ypred),2))
    print()
          
    
    R = {
            'Model': models,
            'accuracy': round(accuracy_score(y_test, ypred), 2),
            'precision': round(precision_score(y_test, ypred), 2),
            'recall': round(recall_score(y_test, ypred), 2),
            'f1': round(f1_score(y_test, ypred), 2)
     }
    
    Results = Results.append(R, ignore_index=True)
    
Results['Accuracy'] = Results['Accuracy'].astype(float)
Results['Precision'] = Results['Precision'].astype(float)
Results['Recall'] = Results['Recall'].astype(float)
Results['f1'] = Results['f1'].astype(float)


# In[52]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# In[53]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[54]:


# ypred = clf.predict(X_test)
# ypred_prob_val = clf.predict_proba(X_val)
ypred_val = clf.predict(X_test)
print(classification_report(y_test, ypred_val))
# ypred_prob_val = clf.predict_proba(X_val)
# print(classification_report(y_test, ypred_prob_val))


# In[55]:


ypred_prob_val = clf.predict_proba(X_test)


# In[56]:


# test_data = pd.read_csv('C:/Users/DAVID/Downloads/Train.csv')
test_data = pd.read_csv('C:/Users/DAVID/Downloads/Test.csv')


# In[57]:


df_test = test_data.drop(['user_id', 'MRG'], axis='columns')
df_test[['ORANGE', 'TIGO', 'ZONE1', 'ZONE2']] = df_test[['ORANGE', 'TIGO', 'ZONE1', 'ZONE2']].fillna(0)
df_test['TOP_PACK'] = df_test['TOP_PACK'].fillna('No Pack')
df_test['FREQ_TOP_PACK'] = df_test['FREQ_TOP_PACK'].fillna(0)
df_test['MONTANT'] = df_test['MONTANT'].fillna(average_montant)
df_test['REGION'] = df_test['REGION'].apply(lambda x: np.random.choice(unique_regions) if pd.isnull(x) else x)
df_test['REVENUE'] = df_test['REVENUE'].fillna(average_revenue)
df_test['ARPU_SEGMENT'] = df_test['ARPU_SEGMENT'].fillna(average_arpu_segment)
df_test['FREQUENCE_RECH'] = df_test['FREQUENCE_RECH'].fillna(average_frequence_rech)
df_test['FREQUENCE'] = df_test['FREQUENCE'].fillna(average_frequence)
df_test['DATA_VOLUME'] = df_test['DATA_VOLUME'].fillna(average_data_volume)
df_test['ON_NET'] = df_test['ON_NET'].fillna(average_on_net)

df_test['TENURE'] = label_encoder.fit_transform(df_test['TENURE'])
df_test['REGION'] = df_test['REGION'].map(average_regularity_per_region)
df_test['TOP_PACK'] = df_test['TOP_PACK'].map(top_pack_freq)
# df_test = df_test.applymap(lambda x: float(x))


# In[58]:


if 'CHURN' in df_test.columns:
    df_test = df_test.drop(['CHURN'], axis=1)


# In[59]:


df_test = df_test.fillna(0)
X_test_final = mms.transform(df_test)


# In[60]:


df_test.isnull().sum()


# In[62]:


# Predict probabilities on test data
ypred_prob_test = clf.predict_proba(X_test_final)


# In[63]:


output_test = pd.DataFrame({'user_id': test_data['user_id'], 'CHURN': ypred_prob_test[:, 1]})
output_test['CHURN'] = output_test['CHURN'].round(4)


# In[64]:


output_test.to_csv('C:/Users/DAVID/Downloads/Test_Output.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




