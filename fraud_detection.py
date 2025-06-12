import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,roc_curve, auc 
import seaborn as sns 
import matplotlib.pyplot as plt 


df = pd.read_csv('/Users/macbookie/Documents/Work/3.Personal/Upskilling/Python/Machine Learning/fraud_detetction_project/data/creditcard_2023.csv')
x = df.drop(['Time','Amount','Class','id'], axis =1, errors='ignore')
y = df['Class']

# Splitting the data into training and testing dataset

# Test size specifies that 20% of the data is used as testing data, 80% is used for training
# Random State ensures we can reproduce our results. 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Feature scaling are applied to ensure features contribute equally to the models learning process
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# We see that the data set is split ±50% as not fraudulent and ±49.99% as fraudulent
print(pd.Series(y_train).value_counts(normalize=True))


# Now to train the model using Random Forrest, combines multiple trees to make robust decisions
rf_model = RandomForestClassifier(
    n_estimators=10, # number of trees in our forrest
    max_depth=10, # number of depth of each tree
    min_samples_split=2, # number of minimum samples required to split
    random_state=42 # reproduce the randomness
)

cv_scores = cross_val_score(rf_model,x_train_scaled,y_train,cv=5,scoring='f1')
rf_model.fit(x_train_scaled, y_train)
y_pred = rf_model.predict(x_test_scaled)

# This creates a visual heatmap of the true postiives and true negatives as well as the false positives and false negatives
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# Top fetaures that have a significant impact - illustrated with a bar graph
importance = rf_model.feature_importances_
feature_imp = pd.DataFrame({
    'Feature':x.columns,
    'Importance': importance
}).sort_values('Importance',ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data = feature_imp,x='Importance', y = 'Feature')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()


# This illustrates the correlation between 
plt.figure(figsize=(12,8))
correlation_matrix=x.corr()
sns.heatmap(correlation_matrix,cmap = 'coolwarm',center=0,annot=True,fmt='.2f') #center is 
plt.title('Feature Correlation Matric')
plt.tight_layout()
# plt.show()


#Fleshing out the roc_curve 
y_pred_proba = rf_model.predict_proba(x_test_scaled)[:,1]
fpr,tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr,tpr)

#Viuslising the roc curve 

plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,color='darkorange',lw=2, label = f'ROC curve  (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],color = 'navy', lw=2, linestyle='--')

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('Reciever Operating Characteristics (ROC) Curve')
plt.legend(loc='lower right')
plt.show()