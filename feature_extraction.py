# BSD 2-Clause License
#
# Copyright (c) [2022], [emre aydoğan], emreaydoan@gmail.com
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os
from pathlib import Path
import sys
import csv
import enum
import shutil
import multiprocessing
from typing import List

import literadar
import smali_parser
import java_parser


# creating enumerations using class
class FeatureType(enum.Enum):
    Apktool = 0
    CustomSmali = 1
    AllSmali = 2
    Permission = 3
    Library = 4
    Metadata = 5
    Java = 6

NEW_FEATURE_LIST = False
def write_feature_list(colnames: List[str], file=Path("./Data/feature_list.meta")):
    '''
    Pass in list of column names to write to a file
    '''
    global NEW_FEATURE_LIST
    f = open(file, "w" if not NEW_FEATURE_LIST else "a")
    f.write(",".join(colnames) + "\n")
    f.close()
    NEW_FEATURE_LIST = True

def parseApk(iron_apk_path, featureTypes, csvname, auth=0, acnt=0):
    print(iron_apk_path, featureTypes, csvname, auth, acnt)
    lrd = literadar.LibRadarLite(iron_apk_path)
    res = lrd.compare()
    print("Dec:", lrd.dec_path)
    apkFeature = []
    vectorIndex = 0
    write_feature_list(["apkname"])

    for featureType in featureTypes:
        ft = int(featureType)
        if ft == FeatureType.CustomSmali.value:
            sp = smali_parser.SmaliFile(lrd.dec_path, 0)
            sf = sp.parseSmali
            sfs = sorted(sf.items(), key=lambda k: k[0])
            for s in sfs:
                # print(s[0])
                vectorIndex += 1
                apkFeature.append(float(s[1]))
            write_feature_list(["customsmali_" + str(i) for i in range(len(sfs))])
        elif ft == FeatureType.Java.value:
            jp = java_parser.JavaFile(lrd.dec_path.replace('Smali', 'Java'), 1)
            jf = jp.parse
            jfs = sorted(jf.items(), key=lambda k: k[0])
            for j in jfs:
                # print(j[0])
                vectorIndex += 1
                apkFeature.append(float(j[1]))
            write_feature_list(["java_" + str(i) for i in range(len(jfs))])
        elif ft == FeatureType.AllSmali.value:
            sp = smali_parser.SmaliFile(lrd.dec_path, 1)
            sf = sp.parseSmali
            sfs = sorted(sf.items(), key=lambda k: k[0])
            for s in sfs:
                # print(s[0])
                vectorIndex += 1
                apkFeature.append(float(s[1]))
            write_feature_list(["allsmali_" + str(i) for i in range(len(sfs))])
        elif ft == FeatureType.Permission.value:
            sp = smali_parser.SmaliFile(lrd.dec_path, 0)
            pf = sp.parsePermission(lrd.dec_path)
            pfs = sorted(pf.items(), key=lambda k: k[0])
            for p in pfs:
                # print(p[0])
                vectorIndex += 1
                apkFeature.append(float(p[1]))
            write_feature_list(["perm_" + str(i) for i in range(len(pfs))])
        elif ft == FeatureType.Library.value:
            sp = smali_parser.SmaliFile(lrd.dec_path, 0)
            lf = sp.getLibrary()
            lfs = sorted(lf.items(), key=lambda k: k[0])
            for l in lfs:
                #        ff.write(l[0] + '\n')
                # print(l[0])
                vectorIndex += 1
                apkFeature.append(float(l[1]))
            write_feature_list(["lib_" + str(i) for i in range(len(lfs))])
    lrd.__close__()
    print(auth, acnt, iron_apk_path)
    return apkFeature


def creator_feature_extract(creator, csv_path, featureTypes, feature_folder, auth=0):
    if not os.listdir(creator):
        return
    csvname = csv_path + '/' + feature_folder + "/" + creator.split('/')[-1] + '.csv'
    # csvname = csv_path + '/' + feature_folder + "/" + feature_folder + '.csv'
    with open(csvname, 'w') as wf:
        writer = csv.writer(wf)
        acnt = 0
        for apk in os.listdir(creator):
            apkpath = creator + '/' + apk
            if apk.endswith(".apk"):
                try:
                    apkf = parseApk(apkpath, featureTypes, csvname, auth, acnt, )
                    if apkf:
                        l = [apk[:-4].replace(',', '_').replace('\'', '_').replace('"', '_')]
                        l += apkf
                        l.append(creator.split("/")[-1])
                        writer.writerow(l)

                except:
                    import traceback
                    print("-------------------------------------")
                    print("PARSE APK ERROR!: " + str(auth), str(acnt), apkpath)
                    traceback.print_exc()
                    print("-------------------------------------")
                    f = open("errors.txt", "w+")
                    sys.stdout.flush()
                acnt += 1


def main():
    MARKET_NAME = sys.argv[1] #'malware'
    APK_FOLDER = os.getcwd() + '/apk'
    DEC_PATH = os.getcwd() + '/smali'
    FEATURE_FOLDER_NAME = os.getcwd() + '/feature'
    creator_path = APK_FOLDER + '/' + MARKET_NAME
    csv_path = FEATURE_FOLDER_NAME + "/" + MARKET_NAME
    apk_number_per_author = 2
    NUMBER_OF_NGRAM = 10000
    pool_size = 8

    featureTypes = sys.argv[2:]
    #featureTypes = [1, 3, 4]
    feature_folder = ""
    for featureType in featureTypes:
        ft = int(featureType)
        if ft == FeatureType.CustomSmali.value:
            feature_folder += "customsmali_"
        elif ft == FeatureType.Java.value:
            feature_folder += "java_"
        elif ft == FeatureType.AllSmali.value:
            feature_folder += "allsmali_"
        elif ft == FeatureType.Permission.value:
            feature_folder += "perm_"
        elif ft == FeatureType.Library.value:
            feature_folder += "lib_"
        elif ft == FeatureType.Apktool.value:
            feature_folder += "apktool_"

    feature_folder = feature_folder[:-1]

    if not os.path.exists(DEC_PATH):
        os.mkdir(DEC_PATH)

    if not os.path.isdir(creator_path):
        print("Can NOT find directory: '%s'" % creator_path)
        exit(1)

    try:
        if os.path.exists(csv_path + '/' + feature_folder):
            shutil.rmtree(csv_path + '/' + feature_folder)
    except:
        pass
    os.makedirs(csv_path + '/' + feature_folder)
    multi_thread = True

    creatorList = []
    dictCreator = {}

    for creator in os.listdir(creator_path):
        apks = creator_path + '/' + creator
        creatorList.append(apks)

        try:
            path, dirs, files = next(os.walk(apks))
            file_count = len(files)
            if file_count >= apk_number_per_author:
                dictCreator[apks] = file_count
        except Exception as e:
            print(e)

    creatorList = []
    sfs = sorted(dictCreator.items(), key=lambda k: k[1], reverse=True)
    for s in sfs:
        creatorList.append(s[0])

    if not multi_thread:
        count = 0
        for creator in creatorList:
            creator_feature_extract(creator, csv_path, featureTypes, feature_folder, count, )
    else:
        print("Processes\t%d" % int(pool_size))
        pool = multiprocessing.Pool(processes=int(pool_size))
        result = []
        count = 0
        for creator in creatorList:
            result.append(
                pool.apply_async(creator_feature_extract, (creator, csv_path, featureTypes, feature_folder, count,)))
            count += 1
        pool.close()
        pool.join()

    csv_folder = csv_path + '/' + feature_folder

    lines = []
    try:
        path, dirs, files = next(os.walk(csv_folder))
        with open('{}/{}.csv'.format(csv_folder, feature_folder), 'w') as wf:
            writer = csv.writer(wf)
            for file in files:
                with open('{}/{}'.format(csv_folder, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        writer.writerow(line.strip().split(','))
    except Exception as e:
        print(e)

    return


if __name__ == '__main__':
    main()
