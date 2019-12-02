import scipy.io
from datetime import datetime, timedelta
import time
import sys, os
import itertools
import numpy
from collections import deque
import numpy as np
import pandas as pd



matlab_filename = 'realitymining.mat'
matlab_obj = scipy.io.loadmat(matlab_filename)

## create df for predicting class labels
affil_y = pd.DataFrame({'id': range(len(matlab_obj['s'][0]['my_affil'])),
                       'affil': matlab_obj['s'][0]['my_affil']
                       })
affil_y['affil_clean'] = np.nan
for j in range(len(affil_y)):
    if len(affil_y.affil[j]) > 0:
        affil_y['affil_clean'][j] = affil_y.affil[j][0][0][0]
affil_y.to_csv('my_affil.csv', index = False)

## create df for converting between id and mac
mac_x = pd.DataFrame({'id':range(len(matlab_obj['s'][0]['mac'])),
                     'mac':matlab_obj['s'][0]['mac']
                     })
mac_x['mac_clean'] = np.nan
for j in range(len(mac_x)):
    if len(mac_x.mac[j]) > 0:
        #print j
        mac_x['mac_clean'][j] = mac_x.mac[j][0][0]
mac_x.to_csv('mac_to_id.csv', index = False)


bt_nw = pd.DataFrame({})
for j in range(len(matlab_obj['s'][0]['mac'])):
    print j
    for time in range(len(matlab_obj['s'][0]['device_date'][j][0])):
        if len(matlab_obj['s'][0]['device_macs'][j][0][time].flatten()) > 0:
            #print range(len(matlab_obj['s'][0]['mac']))[j]
            tmp = pd.DataFrame({'time':np.repeat(matlab_obj['s'][0]['device_date'][j][0][time],len(matlab_obj['s'][0]['device_macs'][j][0][time])),
                               'from_mac':np.repeat(range(len(matlab_obj['s'][0]['mac']))[j],#matlab_obj['s'][0]['mac'][j],
                                                    len(matlab_obj['s'][0]['device_macs'][j][0][time])),
                               'to_mac': matlab_obj['s'][0]['device_macs'][j][0][time].flatten()})
            bt_nw = bt_nw.append(tmp, ignore_index = True)
bt_nw.to_csv('bluetooth_nw.csv', index = False)
