# SourceCodeBasedAAA

1- Download "lite_dataset_10.csv" from "https://github.com/buptkick/AppAuth/tree/master/Data" and put the file to "Data" folder. </br>
2- Create "feature" and "smali" folder in the working directory. </br>
2.1- feature_extraction.py extracts a feature csv file inside "feature" folder according to feature types </br>
2.2- When extracting features, apk files are decompiled the smali codes inside "smali" folder. You can delete these smali codes after feature extraction. </br>
3. The applications must be inside "apk" folder. feature_extraction.py automatically reads all applications inside "apk/$database_name" folder. The structure of the folders must be like below: </br>
    /apk </br>
   -/$database_name </br>
   --/$developer_name </br>
   ---/*.apk </br>

<h1>Feature Extraction</h1> </br>
1- Use Python 2.7 </br>
2- You can use any combination of the following features:/ </br>
class FeatureType(enum.Enum): </br>
&emsp;CustomSmali = 1 </br>
&emsp;AllSmali = 2 </br>
&emsp;Permission = 3 </br>
&emsp;Library = 4 </br>
&emsp;Metadata = 5 </br>
&emsp;Java = 6 </br>
    
3- e.g. use below command to extract customsmali_perm_lib features. This command creates customsmali_perm_lib.csv inside the "feature/$database_name/customsmali_perm_lib/" folder.</br>
```
python feature_extraction.py $database_name 1 3 4
python feature_extraction.py sdkid-creator-path 1 2 3 4 5 6
```

> Notice: For reproductions, `miniconda` is strongly recomended. `mkvirtualenv` has stopped supporting python 2 environments. And ubuntu's default python2.7 do not have pip2, pip is pip3.
>
> Edit: `feature_extraction.py` has been updated to support python3.6. You can use python3.6 to run the script. However this upgrade is somehow incomplete. You need to run the above test cli command, and debug untill it produce ascii-csv files.

> Notice: The defalt ./apk dataset cannot be runned. Set ` apk_number_per_author = 3` to test it.

> Notice: liteRadar in literadar.py uses an unpresented cli tool `baksmali`. This is a google R8 repository now and you need to build a fat jar to enable cli functionalities under jdk11. I am using a old version by JesusFreke, `baksmali-2.5.2.jar`, where cli interface is included in release versions. jdk17 works fine with this jar.
 
> Notice: Some feature sets are conflicting with each other. For example, `xxxsmali` and `java` cannot be extracted at the same time.

<h1>CrossValidation</h1>
1- Use Python 3.6</br>
2- To get result for customsmali_perm_lib.csv, run below command.</br>

```
time python accuracy_results.py $database_name customsmali_perm_lib
```

3- Above command prints a accuracy and time to the console and should be like below:</br>

```
RF acc : 0.9756463453
RF f1 : 0.968457474
```






<h2>Results</h2>
1- Uncomment machine learning models on "classifiers" variable on line 151 in accuracy_analysis.py. You can change or edit models whatever you want. Change the "names" string array variable accordingly.</br>

2- Run below commands

```
python2.7 feature_extraction.py $database_name 1
python3 accuracy_results.py $database_name customsmali


python2.7 feature_extraction.py $database_name 1 4
python3 accuracy_results.py $database_name customsmali_lib

python2.7 feature_extraction.py $database_name 2
python3 accuracy_results.py $database_name allsmali


python2.7 feature_extraction.py $database_name 1 3 4
python3 accuracy_results.py $database_name customsmali_perm_lib

python2.7 feature_extraction.py $database_name 2 3
python3 accuracy_results.py $database_name allsmali_perm
```

3- To add AppAuth features, download source code of AppAuth from "https://github.com/buptkick/AppAuth". The output of feature csv files for AppAuth and our tool are same. You can combine two csv files side by side.</br>

<h2>N-Gram Results</h2>
1- Change the "number_of_ngram" variable in accuracy_analysis.py to number you want. Default number is 10000. Run below command

```
python accuracy_results.py $database_name ngram
```


## References

# Reproduction Settings

## Feature Extraction

```python
apk_number_per_author = 2
```