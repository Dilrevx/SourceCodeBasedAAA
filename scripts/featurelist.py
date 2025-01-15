'''
This script generate ./Data/feature_list file which contains the column names of the feature csv files.
'''

import os

def generate_feature_list():
    '''
    This function generates the feature list file.
    '''
    column_names = ["apkname", "smali", "conf", "res", "dex", "perm", "lib"]
    with open(os.getcwd() + "/Data/feature_list", "w") as f:
        for column in column_names:
            f.write(','.join([column + str(i) for i in range(500)]))
            f.write("\n")
    

if __name__ == "__main__":
    generate_feature_list()