# 9/17/2018
# running locally: goal to try networkx version
#/home/kaltenb/nw_structure_LI_LD/code
#/Users/kristen/Dropbox/monophily_extension/nw_structure_LI_LD/code/www_2019

from __future__ import division
import os
import os
import itertools
import community
from datetime import datetime, timedelta


folder_directory =os.getcwd()
os.chdir(folder_directory)
execfile('../functions/python_libraries.py')
execfile('../functions/parsing.py')  # Sam Way's Code
execfile('../functions/mixing.py')   # Sam Way's Code
execfile('../functions/create_adjacency_matrix.py')
np.seterr(divide='ignore', invalid='ignore')


os.chdir('/Users/kristen/Dropbox/monophily_extension/nw_structure_LI_LD/data/RealityMining/convert_bluetooth_nw')

## set-up RM network
mac_x_id = pd.read_csv('./mac_to_id.csv',dtype={'mac': object})
mac_x_id = mac_x_id.loc[~mac_x_id.mac_clean.isnull()] ## we remove those without mac addresses (id's 104, 105)
bt_nw = pd.read_csv('./bluetooth_nw.csv')
grad_y = pd.read_csv('./my_affil.csv') ## however, id=105 has survey data
## really should be 93 -- but to get 94, we'll keep id=105 since we have survey data,
## but note that id=105 will not be a node in the network because we don't have unit's mac address
ids_subset = np.unique(mac_x_id.id)
ids_survey_complete = np.unique(grad_y.id[~grad_y.affil_clean.isnull()])
## so instead -- we use this set of IDs.
mac_x_id = pd.read_csv('./mac_to_id.csv',dtype={'mac': object})
ids_subset = np.unique(mac_x_id.id)
ids_survey_complete = np.unique(grad_y.id[~grad_y.affil_clean.isnull()])
mac_x_id = mac_x_id.loc[np.in1d(mac_x_id.id, ids_survey_complete)]
grad_y = grad_y[~grad_y.affil_clean.isnull()]
bt_nw = bt_nw.loc[np.in1d(bt_nw.from_mac, mac_x_id.id)]
bt_nw = bt_nw.loc[np.in1d(bt_nw.to_mac, mac_x_id.mac_clean)]
bt_nw.columns = ['from_mac_id','time','to_mac']
    
## merge in from_mac
bt_nw = bt_nw.merge(mac_x_id[['id','mac_clean']],
                    left_on = 'from_mac_id',
                    right_on = 'id',
                    how = 'left')

bt_nw.drop('id', axis = 1, inplace = True)
bt_nw.rename(columns={"mac_clean": "from_mac"}, inplace = True)

bt_nw = bt_nw.merge(mac_x_id[['id','mac_clean']],
                    left_on = 'to_mac',
                    right_on = 'mac_clean',
                    how = 'left')
bt_nw.drop('mac_clean', axis = 1, inplace = True)

bt_nw.rename(columns={"id": "to_mac_id"}, inplace = True)

## https://jeremykun.com/2014/01/21/realitymining-a-case-study-in-the-woes-of-data-processing/
def convertDatetime(dt):
    return datetime.fromordinal(int(dt)) + timedelta(days=dt%1) - timedelta(days=366) - timedelta(hours=5)
        
tmp = map(convertDatetime, bt_nw.time)

month = []
year = []
for j in range(len(tmp)):
    month.append(tmp[j].month)
    year.append(tmp[j].year)


bt_nw['month'] = month
bt_nw['year'] = year
    
##
## Compute LI Features
##
from sklearn.metrics import f1_score

month_yr_train_test = bt_nw[['month','year']].drop_duplicates().sort_values(['year','month'])[3:15]
grad_y[['id','affil_clean']].to_csv('./by_month/labels.csv', index = False)

## break out by month/year
all_mnth_yr = pd.DataFrame() ## additionally create dataframe w all months/years
for j in range(len(month_yr_train_test)):
    month = month_yr_train_test.iloc[j]['month']
    year = month_yr_train_test.iloc[j]['year']
    bt_nw_subset = bt_nw[(bt_nw.month==month) & (bt_nw.year==year)]
    tag = str(month) + '_' + str(year)
    print tag
    ## remove self-loops
    bt_nw_subset = bt_nw_subset.loc[bt_nw_subset.from_mac_id != bt_nw_subset.to_mac_id]
    ## make undirected
    bt_nw_subset_undirect = bt_nw_subset.copy()
    bt_nw_subset_undirect['to_mac_id'] = bt_nw_subset['from_mac_id']
    bt_nw_subset_undirect['from_mac_id'] = bt_nw_subset['to_mac_id']
    
    bt_nw_subset = bt_nw_subset.append(bt_nw_subset_undirect)
    Gnx = nx.from_pandas_edgelist(bt_nw_subset,
                                  source = 'from_mac_id',
                                  target = 'to_mac_id', create_using = nx.DiGraph())
    for edge in Gnx.edges():
        Gnx[edge[0]][edge[1]]['weight']=1
    nx.write_edgelist(Gnx, './by_month/' + tag + '.csv', data = ['weight'], delimiter=',')
    all_mnth_yr = all_mnth_yr.append(bt_nw_subset)



## create nw across all time periods
Gnx_all = nx.from_pandas_edgelist(all_mnth_yr,
                              source = 'from_mac_id',
                              target = 'to_mac_id', create_using = nx.DiGraph())
for edge in Gnx_all.edges():
    Gnx_all[edge[0]][edge[1]]['weight']=1
nx.write_edgelist(Gnx_all, './by_month/all_months_years.csv', data = ['weight'], delimiter=',')
