# About: create edgelists for FB100 dataset
import os
import itertools
import community

folder_directory =os.getcwd()
os.chdir(folder_directory)
execfile('../python_libraries.py')
execfile('../parsing.py')  # Sam Way's Code
execfile('../mixing.py')   # Sam Way's Code
execfile('../create_adjacency_matrix.py')
np.seterr(divide='ignore', invalid='ignore')
fb100_file = '/home/kaltenb/FBdata/data/0_original'
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV


for f in listdir(fb100_file)[25:52]:
    if f.endswith('.mat'):
        tag = f.replace('.mat', '')
        print tag
        input_file = path_join(fb100_file, f)
        A, metadata = parse_fb100_mat_file(input_file)
        adj_matrix_tmp = A.todense()
        gender_y_tmp = metadata[:,1] #gender
        gender_dict = create_dict(range(len(gender_y_tmp)), gender_y_tmp)
        (gender_y, adj_matrix_gender) = create_adj_membership(
                                                              nx.from_numpy_matrix(adj_matrix_tmp), # graph
                                                              gender_dict,   # dictionary
                                                              0,             # val_to_drop, gender = 0 is missing
                                                              'yes',         # delete_na_cols, ie completely remove NA nodes from graph
                                                              0,             # diagonal
                                                              None,          # directed_type
                                                              'gender')      # gender
        gender_y = np.array(map(np.int,gender_y)) ## need np.int for machine precisions reasons
        d = {'keys':range(len(gender_y)),
            'gender_y':gender_y}
        y_df = pd.DataFrame(d)
        output_dir = '/Users/kristen/Dropbox/monophily_extension/nw_structure_LI_LD/code/refex-rolx-master/sample-data/'
        Gnx = nx.from_numpy_matrix(adj_matrix_gender,
                                   create_using=nx.DiGraph())
        fb100_df_tmp = []
        for j in nx.generate_edgelist(Gnx, data = False):
            fb100_df_tmp.append(map(np.int,j.split(' ')))
        fb100_df = pd.DataFrame(fb100_df_tmp)
        fb100_df.columns = ['from','to']
        fb100_df_undirect = fb100_df.copy()
        fb100_df_undirect['to'] = fb100_df['from']
        fb100_df_undirect['from'] = fb100_df['to']
        fb100_df = fb100_df.append(fb100_df_undirect)
        fb100_df['weight'] = 1
        fb100_df.head()
        fb100_df.drop_duplicates(inplace = True)
        ## save output
        fb100_df.to_csv('../../data/FB100edges/'+tag+'.csv', header = False, index = False)
        y_df.to_csv('../../data/FB100edges/'+tag+'_gender.csv', header = True, index = False)
