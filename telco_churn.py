#----------------------------------------------------------------------------------------------------------------------------------------------------------#
# Load and check file
#----------------------------------------------------------------------------------------------------------------------------------------------------------#

file = 'https://raw.githubusercontent.com/theleadio/datascience_demo/master/telco_customer_churn_dataset.csv'

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(5,5)})
sns.set_style('whitegrid')

df = pd.read_csv(file)
df.shape
df.sample()
df.info()

obj = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges', 'Churn']
for column in obj:
  print(column, ':', df[column].unique(), '\n')

df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')

freq_no = len(df[df['Churn'] == 'No'])
freq_yes = len(df[df['Churn'] == 'Yes'])
size = len(df)/100
pd.DataFrame({'Churn': ['No', 'Yes'],
              'Count': [freq_no, freq_yes], 'Proportion (%)': [round(freq_no/size, 2), round(freq_yes/size, 2)]})

sns.histplot(df, x = 'Churn', hue = 'Churn')

label = ['Churn']
binary = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
assume_binary = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
nom = ['gender', 'InternetService', 'Contract', 'PaymentMethod']
cont = ['tenure', 'MonthlyCharges', 'TotalCharges']


#----------------------------------------------------------------------------------------------------------------------------------------------------------#
# Graphical Descriptions
#----------------------------------------------------------------------------------------------------------------------------------------------------------#

cat = binary + assume_binary + nom
fig, ax = plt.subplots(len(cat), 2, figsize = (10, 5*len(cat)), sharey = False)
for i in range(len(cat)):
    cnt = df.groupby(cat[i])['Churn'].value_counts(normalize = False).reset_index(name = 'count')
    sns.histplot(x = cat[i] , hue = 'Churn', weights = 'count', multiple = 'stack', data = cnt, 
                 shrink = 0.8, ax = ax[i,0])
    perc = df.groupby(cat[i])['Churn'].value_counts(normalize = True).mul(100).reset_index(name = 'percentage')
    sns.histplot(x = cat[i] , hue = 'Churn', weights = 'percentage', multiple = 'stack', data = perc, 
                 shrink = 0.8, ax = ax[i,1]).set(ylabel = 'Percentage')
    plt.xticks(rotation = 45)

fig, ax = plt.subplots(len(cont), 2, figsize = (10, 5*len(cont)), sharey = False)
for i in range(len(cont)):
    sns.boxplot(x = df['Churn'], y = df[cont[i]], ax = ax[i,0]);
    sns.histplot(df, x = cont[i], hue = 'Churn', ax = ax[i,1]);

#----------------------------------------------------------------------------------------------------------------------------------------------------------#
# Data preparation for modelling
#----------------------------------------------------------------------------------------------------------------------------------------------------------#

def to_binary(var):
  if var == 'Yes': return 1
  else: return 0
  
everybody = label + binary + assume_binary
for column in everybody:
  df[column] = df[column].apply(to_binary)
  
df = pd.get_dummies(data = df, columns = nom)

sns.set(rc = {'figure.figsize':(20, 18)})
sns.heatmap(df.corr('pearson'), cmap = 'PuOr', fmt = '.2f', annot = True, vmin = -1, vmax = 1, center = 0);

df = df.dropna().reset_index(drop = True)
train_data = df.drop(['customerID', 'Churn', 'PhoneService', 'MultipleLines', 'OnlineBackup', 'DeviceProtection', 
                      'StreamingTV', 'StreamingMovies', 'gender_Male', 'gender_Female', 'PaymentMethod_Mailed check'], axis = 1)
train_label = df['Churn']

#----------------------------------------------------------------------------------------------------------------------------------------------------------#
# Decision Tree Modelling
#----------------------------------------------------------------------------------------------------------------------------------------------------------#

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import graphviz

X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size = 0.2, random_state = 99, stratify = train_label)

model = tree.DecisionTreeClassifier(max_depth = 4)
model.fit(X_train, y_train);
columns = list(train_data.columns)
dot_data = tree.export_graphviz(model, out_file = None, feature_names = columns, class_names = ['No', 'Yes'], filled = True, rounded = True)
y_pred = model.predict(X_test)
graphviz.Source(dot_data)

print(f"Accuracy = {metrics.accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {metrics.precision_score(y_test, y_pred):.4f}")
print(f"Recall = {metrics.recall_score(y_test, y_pred):.4f}")
print(f"F1 Score = {metrics.f1_score(y_test, y_pred):.4f}")


# --->> Balance the dataset with undersampling

import numpy as np

no_churn_index = df[df.Churn == 0].index
random_index = np.random.choice(no_churn_index, 2500, replace = False)
churn_index = df[df.Churn == 1].index
undersample_index = np.concatenate([churn_index, random_index])
under_sample = df.loc[undersample_index]

sns.set(rc={'figure.figsize':(5,5)})
sns.countplot(x = 'Churn', data = under_sample);

train_data = under_sample.drop(['customerID', 'Churn', 'PhoneService', 'MultipleLines', 'OnlineBackup', 'DeviceProtection', 
                      'StreamingTV', 'StreamingMovies', 'gender_Male', 'gender_Female', 'PaymentMethod_Mailed check'], axis = 1)
train_label = under_sample['Churn']
model = tree.DecisionTreeClassifier(max_depth = 4)

X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size = 0.2, random_state = 99, stratify = train_label)

model = tree.DecisionTreeClassifier(max_depth = 4)
model.fit(X_train, y_train);
columns = list(train_data.columns)
dot_data = tree.export_graphviz(model, out_file = None, feature_names = columns, class_names = ['No', 'Yes'], filled = True, rounded = True)
y_pred = model.predict(X_test)
graphviz.Source(dot_data)

print(f"Accuracy = {metrics.accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {metrics.precision_score(y_test, y_pred):.4f}")
print(f"Recall = {metrics.recall_score(y_test, y_pred):.4f}")
print(f"F1 Score = {metrics.f1_score(y_test, y_pred):.4f}")


