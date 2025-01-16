'''
This script generate ./Data/feature_list file which contains the column names of the feature csv files.
'''

import os

def generate_feature_list():
    '''
    This function generates the feature list file.
    Read feature_list.meta to get the number of columns for each feature type.
    '''
    column_names_cols = {"apkname": 1, "smali": 18, "conf":500, "res":500, "dex":500, "perm":158, "lib":500}
    with open(os.getcwd() + "/Data/feature_list", "w") as f:
        for column in column_names_cols:
            f.write(','.join([column + str(i) for i in range(column_names_cols[column])]))
            f.write("\n")
    

if __name__ == "__main__":
    generate_feature_list()