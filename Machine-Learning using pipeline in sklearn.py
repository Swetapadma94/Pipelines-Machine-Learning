#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[4]:


iris=load_iris()


# In[5]:


iris


# In[6]:



X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3,random_state=0)


# # Pipelines Creation
# ## 1. Data Preprocessing by using Standard Scaler
# ## 2. Reduce Dimension using PCA
# ## 3. Apply  Classifier
# 

# In[7]:


pipeline_1=Pipeline([('scaler1',StandardScaler()),
                    ('pca1',PCA(n_components=2)),
                    ('classifier1',LogisticRegression(random_state=0))])


# In[9]:


pipeline_2=Pipeline([('scaler2',StandardScaler()),
                    ('pca2',PCA(n_components=2)),
                    ('classifier2',DecisionTreeClassifier(random_state=0))])


# In[11]:


pipeline_3=Pipeline([('scaler3',StandardScaler()),
                    ('pca3',PCA(n_components=2)),
                    ('classifier3',RandomForestClassifier(random_state=0))])


# In[12]:



## LEts make the list of pipelines
pipelines = [pipeline_1, pipeline_2, pipeline_3]


# In[13]:


best_accuracy=0.0
best_classifier=0
best_pipeline=""


# In[14]:


dict_pipeline={0:'Logistic Regression',1:'Decision Tree',2:'Random Forest'}
for pipe in pipelines:
    pipe.fit(X_train,y_train)


# In[20]:



for i,model in enumerate(pipelines):
    print(i,model)
    


# In[18]:



for i,model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(dict_pipeline[i],model.score(X_test,y_test)))


# In[19]:


for i,model in enumerate(pipelines):
    if model.score(X_test,y_test)>best_accuracy:
        best_accuracy=model.score(X_test,y_test)
        best_pipeline=model
        best_classifier=i
print('Classifier with best accuracy:{}'.format(dict_pipeline[best_classifier]))


# # Pipelines Perform Hyperparameter Tuning Using Grid SearchCV

# In[22]:


from sklearn.model_selection import GridSearchCV


# In[23]:


# Create a pipeline
pipe = Pipeline([("classifier", RandomForestClassifier())])
# Create dictionary with candidate learning algorithms and their hyperparameters
grid_param = [
                {"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l2','l1'],
                 "classifier__C": np.logspace(0, 4, 10)
                 },
                {"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l2'],
                 "classifier__C": np.logspace(0, 4, 10),
                 "classifier__solver":['newton-cg','saga','sag','liblinear'] ##This solvers don't allow L1 penalty
                 },
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 100, 1000],
                 
                 "classifier__max_depth":[5,8,15,25,30,None],
                 "classifier__min_samples_leaf":[1,2,5,10,15,100],
                 "classifier__max_leaf_nodes": [2, 5,10]}]
# create a gridsearch of the pipeline, the fit the best model
gridsearch = GridSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1) # Fit grid search
best_model = gridsearch.fit(X_train,y_train)


# In[24]:


print(best_model.best_estimator_)
print("The mean accuracy of the model is:",best_model.score(X_test,y_test))


# # MakePipelines In SKLearn

# In[25]:



from sklearn.pipeline import make_pipeline


# In[26]:



# Create a pipeline
pipe = make_pipeline((RandomForestClassifier()))
# Create dictionary with candidate learning algorithms and their hyperparameters
grid_param = [
                {"randomforestclassifier": [RandomForestClassifier()],
                 "randomforestclassifier__n_estimators": [10, 100, 1000],
                 "randomforestclassifier__max_depth":[5,8,15,25,30,None],
                 "randomforestclassifier__min_samples_leaf":[1,2,5,10,15,100],
                 "randomforestclassifier__max_leaf_nodes": [2, 5,10]}]
# create a gridsearch of the pipeline, the fit the best model
gridsearch = GridSearchCV(pipe, grid_param, cv=5, verbose=0,n_jobs=-1) # Fit grid search
best_model = gridsearch.fit(X_train,y_train)


# In[27]:



best_model.score(X_test,y_test)


# In[ ]:




