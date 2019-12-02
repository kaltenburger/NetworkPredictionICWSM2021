#10/21/2019
from __future__ import division
import os
import re
import itertools
import community
from datetime import datetime, timedelta
from sklearn import preprocessing
import imblearn
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
folder_directory =os.getcwd()
os.chdir(folder_directory)
execfile('python_libraries.py')
execfile('LINK.py')
execfile('parsing.py')  # Sam Way's Code
execfile('mixing.py')   # Sam Way's Code
execfile('create_adjacency_matrix.py')
np.seterr(divide='ignore', invalid='ignore')
y_predict_type = 'gender' # gender
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


percent_initially_unlabelled = [0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05]
percent_initially_labelled = np.subtract(1, percent_initially_unlabelled)

school = 'Amherst41.csv' 

## 3 iterations
x3 = pd.read_csv('../../code/refex-rolx-master-3/sample-data/FB-data/out_union_0.50_FB_'+school+'-featureValues.csv', 
                                  header = -1)

df_train = pd.DataFrame({'keys':np.array(map(np.int,x3[0]))})
x3 = x3.loc[:,1::] #drop IDs column
scaler.fit(x3) #transform each feature on [0,1]-scale
x3 = scaler.transform(x3)


y_train_y = pd.read_csv('../../data/FB100edges/gender_student_'+re.sub('\\.csv','',school) + '_gender.csv')
y3 = df_train.merge(y_train_y[['keys','gender_y']], ## will be in same order as features
        how = 'left',
        on = 'keys')

print(len(y_train_y))


np.unique(y_train_y.gender_y)
len(y_train_y.gender_y)


print np.min(np.unique(df_train['keys']))
print np.max(np.unique(df_train['keys']))
#print len(df_train['keys'])

print ''
print np.min(y_train_y['keys'])
print np.max(y_train_y['keys'])
print len(y3)

## all iterations
xall = pd.read_csv('../../code/refex-rolx-master/sample-data/FB-data/out_union_0.50_FB_'+school+'-featureValues.csv', 
                                  header = -1)

df_train = pd.DataFrame({'keys':np.array(map(np.int,xall[0]))})
xall = xall.loc[:,1::] #drop IDs column
scaler.fit(xall) #transform each feature on [0,1]-scale
xall = scaler.transform(xall)


y_train_y = pd.read_csv('../../data/FB100edges/gender_student_'+re.sub('\\.csv','',school) + '_gender.csv')


yall = df_train.merge(y_train_y[['keys','gender_y', 'year_y']],
        how = 'left',
        on = 'keys')

if y_predict_type=='student':
    yall = yall.loc[(yall.year_y>=2005) & (yall.year_y<=2008)]
    idxall =  np.array(np.where((yall.year_y>=2005) & (yall.year_y<=2008))[0])
    yall.year_y = (yall.year_y >=2007)+0


## 1x iteration
x1 = pd.read_csv('../../code/refex-rolx-master-1/sample-data/FB-data/out_union_0.50_FB_'+school+'-featureValues.csv', 
                                  header = -1)

df_train = pd.DataFrame({'keys':np.array(map(np.int,x1[0]))})
x1 = x1.loc[:,1::] #drop IDs column
scaler.fit(x1) #transform each feature on [0,1]-scale
x1 = scaler.transform(x1)
y_train_y = pd.read_csv('../../data/FB100edges/gender_student_'+re.sub('\\.csv','',school) + '_gender.csv')




y1 = df_train.merge(y_train_y[['keys','gender_y','year_y']],
        how = 'left',
        on = 'keys')


## 2 iterations
x2 = pd.read_csv('../../code/refex-rolx-master-2/sample-data/FB-data/out_union_0.50_FB_'+school+'-featureValues.csv', 
                                  header = -1)

df_train = pd.DataFrame({'keys':np.array(map(np.int,x2[0]))})
x2 = x2.loc[:,1::] #drop IDs column
scaler.fit(x2) #transform each feature on [0,1]-scale
x2 = scaler.transform(x2)
y_train_y = pd.read_csv('../../data/FB100edges/gender_student_'+re.sub('\\.csv','',school) + '_gender.csv')
y2 = df_train.merge(y_train_y[['keys','gender_y','year_y']],
        how = 'left',
        on = 'keys')


print(len(y_train_y))


import imblearn
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[32]:


RF = True # else continues using LogForest models from earlier
undersampling = False

if RF:
    if undersampling:
        ## add in -- undersampling possibly
        max_depth = [3, 5, 10]
        max_depth.append(None)
        min_samples_leaf = [0.05, 0.1, 0.2]
        min_samples_split = [2, 3, 4, 5, 10]
        n_estimators = [50, 100, 150] #[10, 50, 100, 150, 200]
        max_features = ['auto', 0.25, 0.5, 0.75]
        random_grid = {'clf__max_depth': max_depth,
                     'clf__min_samples_leaf': min_samples_leaf,
                     'clf__max_features': max_features,
                      'clf__n_estimators': n_estimators,
                      'clf__min_samples_split': min_samples_split
        }

        clf = sklearn.ensemble.RandomForestClassifier()
        pipeline = imbPipeline([('undersample', imblearn.under_sampling.RandomUnderSampler(random_state=567)),
                                    #('oversample',imblearn.over_sampling.SMOTE()), 
                            ('clf',clf)])
        model1 = RandomizedSearchCV(estimator = pipeline,
                param_distributions = random_grid,
                cv = 3, verbose=0,
                n_jobs = 2)

        model2 = RandomizedSearchCV(estimator = pipeline,
                param_distributions = random_grid,
                cv = 3, verbose=0,
                n_jobs = 2)

        model3 = RandomizedSearchCV(estimator = pipeline,
                param_distributions = random_grid,
                cv = 3, verbose=0,
                n_jobs = 2)

        modelall = RandomizedSearchCV(estimator = pipeline,
                param_distributions = random_grid,
                cv = 3, verbose=0,
                n_jobs = 2)
    else:
        max_depth = [3, 5, 10]
        max_depth.append(None)
        min_samples_leaf = [1,5]#[0.05, 0.1, 0.2]
        min_samples_split = [2,5] #[2, 3, 4, 5, 10]
        n_estimators = [50, 100, 1000]#150]
        max_features = ['auto', 0.25, 0.5, 0.75]
        undersampling_count_arr =  np.array(range(np.sum(10),np.int(np.round(np.sum(np.array(yall.gender_y)==1)*9/10*2/3))+1,
                                                  np.int((np.int(np.round(np.sum(np.array(yall.gender_y)==1)*9/10*2/3))+1 - 10)/25)))
        random_grid = {'max_depth': max_depth,
                        'min_samples_leaf': min_samples_leaf,
                        'max_features': max_features,
                        'n_estimators': n_estimators,
                        'min_samples_split': min_samples_split,
                       'class_weight': [{1:0.505, 1:50.5},{1:0.51, 1:25.5},{1:0.525, 1:10.5},{1:0.55, 1:5.5},{1:0.7, 1:1.75}]
                        }
        
        
        clf = sklearn.ensemble.RandomForestClassifier()
        n = 35
        model1 = RandomizedSearchCV(estimator = clf,
                                   param_distributions = random_grid,
                                   cv = 3, verbose=0, n_iter = n,
                                   n_jobs = 9)

        model2 = RandomizedSearchCV(estimator = clf,
                                    param_distributions = random_grid,
                                    cv = 3, verbose=0, n_iter = n,
                                    n_jobs = 9)
        model3 = RandomizedSearchCV(estimator = clf,
                                    param_distributions = random_grid,
                                    cv = 3, verbose=0, n_iter = n,
                                    n_jobs = 9)
        modelall = RandomizedSearchCV(estimator = clf,
                                    param_distributions = random_grid,
                                    cv = 3, verbose=0, n_iter = n,
                                    n_jobs = 9)

from sklearn.ensemble import BaggingClassifier

num_features = np.shape(np.matrix(xall))[1]
model_LF = BaggingClassifier(linear_model.LogisticRegression(penalty='l2',
                        solver='lbfgs',
                        C=10e20),
                        n_estimators=500,
                        max_features=np.int(np.round(np.log(num_features)))+1)

(mean_accuracy_xall_LF, se_accuracy_xall_LF, 
 mean_micro_auc_xall_LF,se_micro_auc_xall_LF, mean_wt_auc_xall_lbfgs_LF, se_wt_auc_xall_LF)= LINK(percent_initially_unlabelled, ## note: mean_se_model assumes a vector of x% initially labeled
                                                              np.array(yall.gender_y), ## gender labels 
                                                              np.matrix(xall), ## adjacency matrix
                                                              clf = model_LF,
                                                            num_iter=25,
                                                            cv_setup = 'stratified')


if y_predict_type=='gender':
    (mean_accuracy_xall, se_accuracy_xall, 
     mean_micro_auc_xall,se_micro_auc_xall, mean_wt_auc_xall_lbfgs,se_wt_auc_xall)= LINK(percent_initially_unlabelled, ## note: mean_se_model assumes a vector of x% initially labeled
                                                                  np.array(yall.gender_y), ## gender labels 
                                                                  np.matrix(xall), ## adjacency matrix
                                                                  clf = modelall,
                                                                                         num_iter=25,
                                                                                 cv_setup = 'stratified')


if y_predict_type=='gender':
    (mean_accuracy_x3, se_accuracy_x3, 
     mean_micro_auc_x3,se_micro_auc_x3, mean_wt_auc_x3_lbfgs,se_wt_auc_x3)= LINK(percent_initially_unlabelled, ## note: mean_se_model assumes a vector of x% initially labeled
                                                                  np.array(y3.gender_y), ## gender labels 
                                                                  np.matrix(x3), ## adjacency matrix
                                                                  clf = model3,num_iter=25,
                                                                                 cv_setup = 'stratified')


if y_predict_type=='gender':
    (mean_accuracy_x1, se_accuracy_x1, 
     mean_micro_auc_x1,se_micro_auc_x1, mean_wt_auc_x1_lbfgs,se_wt_auc_x1)= LINK(percent_initially_unlabelled, ## note: mean_se_model assumes a vector of x% initially labeled
                                                                  np.array(y1.gender_y), ## gender labels 
                                                                  np.matrix(x1), ## adjacency matrix
                                                                  clf = model1,num_iter=25,
                                                                                 cv_setup = 'stratified')

    




if y_predict_type == 'gender':
    (mean_accuracy_x2, se_accuracy_x2, 
     mean_micro_auc_x2,se_micro_auc_x2, mean_wt_auc_x2_lbfgs,se_wt_auc_x2)= LINK(percent_initially_unlabelled, ## note: mean_se_model assumes a vector of x% initially labeled
                                                                  np.array(y2.gender_y), ## gender labels 
                                                                  np.matrix(x2), ## adjacency matrix
                                                                  clf = model2,num_iter=25,
                                                                                 cv_setup = 'stratified')



edges = pd.read_csv('../../data/FB100edges/'+school, header = -1)
edges.head()
df = pd.crosstab(edges[0], edges[1])
print df.head()
idx = df.columns.union(df.index)
df = df.reindex(index = idx, columns=idx, fill_value=0)

if y_predict_type == 'student':
    df = df.iloc[idxall]


# In[ ]:


if y_predict_type == 'gender':
    (mean_accuracy_LINK_RM, se_accuracy_LINK_RM, 
     mean_micro_auc_LINK_RM,se_micro_auc_LINK_RM, mean_wt_LINK_RM,se_wt_LINK_RM)= LINK(percent_initially_unlabelled, ## note: mean_se_model assumes a vector of x% initially labeled
                                                                  np.array(y_train_y.gender_y), ## gender labels 
                                                                  np.matrix(df), ## adjacency matrix
                                                                  clf = linear_model.LogisticRegression(penalty='l2',
                                                                                                        solver='lbfgs',
                                                                                                        C=10e20),
                                                                num_iter=25, cv_setup = 'stratified')

from matplotlib.backends.backend_pdf import PdfPages


n2vec_amherst = pd.read_csv('../node2vec/emb/Amherst41_dimension32.emb',
                    skiprows=1,
                    header = None,
                    sep=' ')
n2vec_amherst.head()


## merge in gender
tmp = pd.DataFrame(n2vec_amherst[0])
tmp.columns = ['keys']

y_labels = tmp.merge(y_train_y[['keys','gender_y']],
              left_on = 'keys',
              right_on = 'keys',
              how = 'left')
np.sum(y_labels['keys']!=n2vec_amherst[0])

np.unique(y_labels.gender_y)
n2vec_amherst.drop(0, axis =1, inplace = True)

C_vals = [10**x for x in range(-10,100)]



# In[ ]:


(mean_accuracy_node2vec, se_accuracy_node2vec, 
 mean_micro_auc_node2vec,se_micro_auc_node2vec, mean_wt_node2vec,se_wt_node2vec)= LINK(percent_initially_unlabelled, ## note: mean_se_model assumes a vector of x% initially labeled
                                                             # np.array(y_labels.gender_y).astype(np.int), ## gender labels 
                                                            np.array(y_labels.gender_y).astype(np.int)==2,
                                                              np.matrix(n2vec_amherst), ## adjacency matrix
                                            clf = modelall,


                                                                    num_iter=25, 
                                                                     cv_setup = 'stratified')

# Save Results as DataFrame
df_results = pd.DataFrame({'mean_wt_auc_x1_lbfgs': mean_wt_auc_x1_lbfgs,
                            'se_wt_auc_x1': se_wt_auc_x1,
                            'mean_wt_auc_x2_lbfgs': mean_wt_auc_x2_lbfgs,
                            'se_wt_auc_x2': se_wt_auc_x2,
                            'mean_wt_auc_x3_lbfgs': mean_wt_auc_x3_lbfgs,
                            'se_wt_auc_x3': se_wt_auc_x3,
                            'mean_wt_auc_xall_lbfgs': mean_wt_auc_xall_lbfgs,
                            'se_wt_auc_xall': se_wt_auc_xall,
                           'mean_wt_LINK_RM':mean_wt_LINK_RM,
                           'se_wt_LINK_RM': se_wt_LINK_RM,
                           'mean_wt_node2vec':mean_wt_node2vec,
                           'se_wt_node2vec':se_wt_node2vec,
                           'mean_wt_auc_xall_lbfgs_LF':mean_wt_auc_xall_lbfgs_LF,
                           'se_wt_auc_xall_LF':se_wt_auc_xall_LF
                            })
df_results.to_csv('NEWFb100Results.csv',sep=',', index = False)

