import warnings
warnings.filterwarnings('ignore')
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.gridspec import GridSpec
# import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn import metrics
from joblib import dump, load
from utils import cleanResume

#Loading Data
resumeDataSet = pd.read_csv('./data/UpdatedResumeDataSet.csv' ,encoding='utf-8')

# plt.figure(figsize=(15,15))
# plt.xticks(rotation=90)
# sns.countplot(y="Category", data=resumeDataSet)
# plt.savefig('./output/jobcategory_details.png')

# #Pie-chart
# targetCounts = resumeDataSet['Category'].value_counts().reset_index()['Category']
# targetLabels  = resumeDataSet['Category'].value_counts().reset_index()['index']

# # Make square figures and axes
# plt.figure(1, figsize=(25,25))
# the_grid = GridSpec(2, 2)
# plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')
# source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, )
# plt.savefig('./output/category_dist.png')

resumeDataSet['cleaned_resume'] = resumeDataSet['Resume'].apply(lambda x: cleanResume(x))
var_mod = ['Category']

le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])
requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

#Model Building
X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

# print('Model saved.')

# #Results
# print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
# print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
# print("n Classification report for classifier %s:n%sn" % (clf, metrics.classification_report(y_test, prediction)))