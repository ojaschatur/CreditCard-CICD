# Import necessary libraries
import os
import base64
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

# Decode the Base64 encoded Google credentials and write them to a file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-credentials.json'
with open('google-credentials.json', 'w') as file:
    file.write(base64.b64decode(os.environ['GOOGLE_CREDENTIALS']).decode())

# Fetch the dataset from BigQuery
client = bigquery.Client()
query = """
    SELECT * 
    FROM `gitlab-vertexai.gitlab-vertexai.gitlab.creditcard`
"""
query_job = client.query(query)
data = query_job.to_dataframe()

# Preprocessing
y = data['Class']  
X = data.drop(['Class', 'Time'], axis=1)  # Drop 'Time' column if it's not relevant to the model

# Scale 'Amount' column
sc = StandardScaler()
X['Amount'] = sc.fit_transform(X['Amount'].values.reshape(-1,1))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE for imbalanced data
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_res, y_train_res)

# Evaluation
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Save Model
model_filename = "model.pkl"
pickle.dump(model, open(model_filename, 'wb'))

# Upload the model to Google Cloud Storage
storage_client = storage.Client()
bucket = storage_client.get_bucket('gitlab-vertexai-bucket')
blob_model = bucket.blob('models/model.pkl')
blob_model.upload_from_filename('model.pkl')



#Added comment to test branching