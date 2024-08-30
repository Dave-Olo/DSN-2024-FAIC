#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import joblib


# In[2]:


def preprocess_data(df, is_train=True, params=None):
    if is_train:
        # Calculate means (rounded) and modes
        average_orange = round(df['ORANGE'].mean())
        average_tigo = round(df['TIGO'].mean())
        mode_zone1 = df['ZONE1'].mode()[0]
        mode_zone2 = df['ZONE2'].mode()[0]

        # Fill NaN values for the specified columns
        df['ORANGE'] = df['ORANGE'].fillna(average_orange)
        df['TIGO'] = df['TIGO'].fillna(average_tigo)
        df['ZONE1'] = df['ZONE1'].fillna(mode_zone1)
        df['ZONE2'] = df['ZONE2'].fillna(mode_zone2)

        # Fill NaN values for other columns
        df['TOP_PACK'] = df['TOP_PACK'].fillna('No Pack')
        df['FREQ_TOP_PACK'] = df['FREQ_TOP_PACK'].fillna(0)

        unique_regions = df['REGION'].dropna().unique()
        df['REGION'] = df['REGION'].apply(lambda x: np.random.choice(unique_regions) if pd.isnull(x) else x)
        
        average_montant = df['MONTANT'].mean()
        df['MONTANT'] = df['MONTANT'].fillna(average_montant)

        average_revenue = df['REVENUE'].mean()
        df['REVENUE'] = df['REVENUE'].fillna(average_revenue)

        average_arpu_segment = df['ARPU_SEGMENT'].mean()
        df['ARPU_SEGMENT'] = df['ARPU_SEGMENT'].fillna(average_arpu_segment)

        average_frequence_rech = df['FREQUENCE_RECH'].mean()
        df['FREQUENCE_RECH'] = df['FREQUENCE_RECH'].fillna(average_frequence_rech)

        average_frequence = df['FREQUENCE'].mean()
        df['FREQUENCE'] = df['FREQUENCE'].fillna(average_frequence)

        average_data_volume = df['DATA_VOLUME'].mean()
        df['DATA_VOLUME'] = df['DATA_VOLUME'].fillna(average_data_volume)

        average_on_net = df['ON_NET'].mean()
        df['ON_NET'] = df['ON_NET'].fillna(average_on_net)

        # Modify non-numerical parameters
        tenure_mapping = {
            'D 3-6 month': 1,
            'E 6-9 month': 2,
            'F 9-12 month': 3,
            'G 12-15 month': 4,
            'H 15-18 month': 5,
            'I 18-21 month': 6,
            'J 21-24 month': 7,
            'K > 24 month': 8
        }
        df['TENURE'] = df['TENURE'].map(tenure_mapping)
        
        average_regularity_per_region = df.groupby('REGION')['REGULARITY'].mean()
        df['REGION'] = df['REGION'].map(average_regularity_per_region)

        top_pack_freq = df.groupby('TOP_PACK')['FREQ_TOP_PACK'].mean()
        df['TOP_PACK'] = df['TOP_PACK'].map(top_pack_freq)

        # Save mappings and averages
        params = {
            'tenure_mapping': tenure_mapping, 
            'average_montant': average_montant, 
            'average_revenue': average_revenue, 
            'average_arpu_segment': average_arpu_segment, 
            'average_frequence_rech': average_frequence_rech, 
            'average_frequence': average_frequence, 
            'average_data_volume': average_data_volume, 
            'average_on_net': average_on_net,
            'average_orange': average_orange, 
            'average_tigo': average_tigo, 
            'mode_zone1': mode_zone1, 
            'mode_zone2': mode_zone2, 
            'top_pack_freq': top_pack_freq,
            'unique_regions': unique_regions,
            'average_regularity_per_region': average_regularity_per_region
        }
        joblib.dump(params, 'preprocessing_params.pkl')

    else:
        # Load mappings and averages
        tenure_mapping = params['tenure_mapping']
        average_montant = params['average_montant']
        average_revenue = params['average_revenue']
        average_arpu_segment = params['average_arpu_segment']
        average_frequence_rech = params['average_frequence_rech']
        average_frequence = params['average_frequence']
        average_data_volume = params['average_data_volume']
        average_on_net = params['average_on_net']
        average_orange = params['average_orange']
        average_tigo = params['average_tigo']
        mode_zone1 = params['mode_zone1']
        mode_zone2 = params['mode_zone2']
        top_pack_freq = params['top_pack_freq']
        unique_regions = params['unique_regions']
        average_regularity_per_region = params['average_regularity_per_region']

        # Fill NaN values using the loaded parameters
        df['ORANGE'] = df['ORANGE'].fillna(average_orange)
        df['TIGO'] = df['TIGO'].fillna(average_tigo)
        df['ZONE1'] = df['ZONE1'].fillna(mode_zone1)
        df['ZONE2'] = df['ZONE2'].fillna(mode_zone2)
        df['TOP_PACK'] = df['TOP_PACK'].fillna('No Pack')
        df['FREQ_TOP_PACK'] = df['FREQ_TOP_PACK'].fillna(0)
        df['MONTANT'] = df['MONTANT'].fillna(average_montant)
        df['REGION'] = df['REGION'].apply(lambda x: np.random.choice(unique_regions) if pd.isnull(x) else x)
        df['REVENUE'] = df['REVENUE'].fillna(average_revenue)
        df['ARPU_SEGMENT'] = df['ARPU_SEGMENT'].fillna(average_arpu_segment)
        df['FREQUENCE_RECH'] = df['FREQUENCE_RECH'].fillna(average_frequence_rech)
        df['FREQUENCE'] = df['FREQUENCE'].fillna(average_frequence)
        df['DATA_VOLUME'] = df['DATA_VOLUME'].fillna(average_data_volume)
        df['ON_NET'] = df['ON_NET'].fillna(average_on_net)

        # Modify non-numerical parameters
        df['TENURE'] = df['TENURE'].map(tenure_mapping)
        df['TOP_PACK'] = df['TOP_PACK'].map(top_pack_freq)
        df['REGION'] = df['REGION'].map(average_regularity_per_region)

    return df


# In[3]:


# Load and preprocess the training data
df_train = pd.read_csv('C:/Users/DAVID/Downloads/Train.csv')
df_train = preprocess_data(df_train, is_train=True)


# In[4]:


# Drop columns for model training
df_train = df_train.drop(['user_id', 'MRG'], axis=1)


# In[5]:


# Split data into features and target
X = df_train.drop(['CHURN'], axis=1)
y = df_train['CHURN']


# In[6]:


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix


# In[7]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Initializing and training the CatBoostClassifier
classifier = CatBoostClassifier(loss_function='Logloss', learning_rate=0.01, use_best_model=True, eval_metric='Logloss')
classifier.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100, verbose=100)


# In[9]:


model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=0)
model.fit(X_train, y_train)


# In[10]:


# Save the model
joblib.dump(model, 'churn2_cat_model.pkl')


# In[11]:


df_test = pd.read_csv('C:/Users/DAVID/Downloads/Test.csv')


# In[12]:


params = joblib.load('preprocessing_params.pkl')


# In[13]:


df_test = preprocess_data(df_test, is_train=False, params=params)


# In[14]:


# Drop unnecessary columns from the test data
X_test = df_test.drop(['user_id', 'MRG'], axis=1)


# In[15]:


# Load the model
model = joblib.load('churn2_cat_model.pkl')


# In[16]:


# Predict probabilities
predictions_proba = model.predict_proba(X_test)[:, 1]


# In[17]:


predictions_proba = [round(prob, 4) for prob in predictions_proba]


# In[18]:


results = pd.DataFrame({'user_id': df_test['user_id'], 'CHURN': predictions_proba})


# In[19]:


# Save the results to a CSV file
results.to_csv('churn2_cat_predictions.csv', index=False)


# In[20]:





# In[21]:





# In[ ]:




