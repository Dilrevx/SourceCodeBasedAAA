import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class TFPNCounter:
    """
    Calculate True Positive, True Negative, False Positive, False Negative.

    TP: 关联关系是对的
    TN: 本身没有关联关系的 app 没有关联到其他 app
    FP：关联结果错的
    FN：关联关系漏掉的

    AAA 没有 TN，因为一定会输出一个标签
    Negatives: 本身没有关联的 app，
    """

    def __init__(self, dbpth: Path, isNegative: Callable[[str, set[str]], bool]):
        """
        isNegative: 判断是否是 negative 的函数
        """
        self.dbpth = dbpth
        self._isNegative = isNegative

        self.author_apks: Dict[str, set[str]] = {}  # 作者名下的 app 集合
        self._init_author_apks()

        self.negatives: Set[str] = (
            set()
        )  # 作者名下只有一个 app 时，无关联，认为是 negative
        self._init_negatives(isNegative)

    def _init_author_apks(self):
        """
        Read the db directory, get the author name and the corresponding apks.
        """
        for author in self.dbpth.iterdir():
            if author.is_dir():
                apks = set(apk.stem for apk in author.iterdir() if apk.is_file())
                self.author_apks[author.stem] = apks

    def _init_negatives(self, isNegative: Callable[[str, set[str]], bool]):
        """
        isNegative: 判断是否是 negative 的函数

        Iterate over db, update self.negatives with `isNegative` is True.
        """
        self.negatives.update(
            author
            for author, apks in self.author_apks.items()
            if isNegative(author, apks)
        )
        print(f"Negatives: {self.negatives}")

    def count_TP_FP_TN_FN(
        self, y_true: NDArray[np.str_], y_pred: NDArray[np.str_]
    ) -> Tuple[int, int, int, int]:
        """
        y_true: 真实标签
        y_pred: 预测标签
        """
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                if y_true[i] in self.negatives:
                    TN += 1
                else:
                    TP += 1
            else:
                if y_true[i] in self.negatives:
                    FP += 1
                else:
                    FN += 1
        return TP, FP, TN, FN


class AssociationOutputWriter:
    """
    Write the association output to a file. (json)
    """

    def __init__(self, out_folder: Path, author_apks: Dict[str, set[str]]):
        out_folder.mkdir(parents=True, exist_ok=True)
        self.pred_fout = out_folder / "pred.json"
        self.trainset_fout = out_folder / "trainset.json"
        self.testset_fout = out_folder / "testset.json"

        self.author_apks = author_apks

    def dump_trainset(self, dataset: pd.DataFrame):
        ret = defaultdict(list)

        for i in range(dataset.shape[0]):
            author = dataset.iloc[i, -1]
            apkName = dataset.iloc[i, 0]
            ret[author].append(apkName)

        json.dump(ret, self.trainset_fout.open("w"), indent=4)

    def dump_testset(self, dataset: pd.DataFrame):
        ret = defaultdict(list)

        for i in range(dataset.shape[0]):
            author = dataset.iloc[i, -1]
            apkName = dataset.iloc[i, 0]
            ret[author].append(apkName)

        json.dump(ret, self.testset_fout.open("w"), indent=4)

    def dump_pred(self, dataset: pd.DataFrame, y_pred: NDArray, _print=False):
        """
        X: apkName
        y: true label
        y_pred: predicted label
        """
        ret = defaultdict(lambda: defaultdict(list))

        for i in range(dataset.shape[0]):
            author = dataset.iloc[i, -1]
            apkName = dataset.iloc[i, 0]
            pred_author = y_pred[i]

            # ret[author]["ground_truth"].append(apkName)
            ret[pred_author]["predicted"].append(apkName)

            if _print:
                print(f"{apkName} - y-true: {author} -> y-pred {pred_author}")

        # include all authors and apks
        for author, apks in self.author_apks.items():
            ret[author]["ground_truth"] = list(apks)
        json.dump(ret, self.pred_fout.open("w"), indent=4)


def create_dataset_object(feature_type: str, number_of_ngram: int):
    """
    Example: feature_type = 'customsmali_perm_lib', number_of_ngram = 10000

    This function creates a pandas DataFrame object.

    The csv files in `./feature/` do not have column names. Each apk's feature is stored in a row of hundreds of columns.

    /Data/feature_list is used to get the column names.
    Pandas allows the specified names(feature_list content) to be more than the actual number of columns in the DataFrame.
    """
    column_names: List[str] = (
        []
    )  # ["apkName", "smali", "java", "conf", "res", "dex", "perm", "lib", "metadata", "ngram", "label"]
    with open(os.getcwd() + "/Data/feature_list", "r") as f:
        lines = f.readlines()
        column_names.append("apkName")

        types = feature_type.split("_")
        for type in types:
            type = type.split("/")[0]
            if "smali" in type:
                column_names += lines[1].strip().split(",")
            elif "java" in type:
                column_names += lines[1].strip().split(",")
            elif "conf" in type:
                column_names += lines[2].strip().split(",")
            elif "res" in type:
                column_names += lines[3].strip().split(",")
            elif "dex" in type:
                column_names += lines[4].strip().split(",")
            elif "perm" in type:
                column_names += lines[5].strip().split(",")
            elif "lib" in type:
                column_names += lines[6].strip().split(",")
            elif "metadata" in type:
                column_names += ",".join(
                    ["metadata" + str(i) for i in range(50000)]
                ).split(",")
            elif "ngram" in type:
                column_names += ",".join(
                    ["ngram" + str(i) for i in range(number_of_ngram)]
                ).split(",")
            elif "all" in type:
                column_names += ",".join(
                    ["ngram" + str(i) for i in range(number_of_ngram)]
                ).split(",")
            elif "dex" in type:
                column_names += ",".join(
                    ["ngram" + str(i) for i in range(number_of_ngram)]
                ).split(",")
            elif "app" in type:
                column_names += ",".join(
                    ["ngram" + str(i) for i in range(number_of_ngram)]
                ).split(",")
            # elif type == 'all':
            #     column_names += lines[1].strip().split(',')
            #     column_names += lines[2].strip().split(',')
            #     column_names += lines[3].strip().split(',')
            #     column_names += lines[4].strip().split(',')
            #     column_names += lines[5].strip().split(',')
            #     column_names += lines[6].strip().split(',')

        column_names.append("label")
    return pd.DataFrame(columns=column_names)


def read_all_csv_from_db_folder(feature_folder: str, type: str, number_of_ngram: int):
    """
    Example: feature_folder = 'feature/malware/customsmali_perm_lib', type = 'customsmali_perm_lib', number_of_ngram = 10000
    """
    dataset = create_dataset_object(type, number_of_ngram)
    i = 0
    file_name = "{}/{}.csv".format(feature_folder, type)

    with open(file_name, "r") as f:
        data = pd.read_csv(f, header=None, names=dataset.columns)
        dataset = pd.concat([dataset, data], ignore_index=True)
        i += 1
    return dataset


def combine_dataset(X, y, apkName: List, column_list):
    """
    Inverse of split_dataset function.
    (X, y, apkName) -> dataset
    """
    temp = np.concatenate((np.array([np.array(apkName)]).T, X), axis=1)
    temp = np.concatenate((temp, np.array([np.array(y)]).T), axis=1)
    return pd.DataFrame(data=temp, index=range(0, len(temp)), columns=column_list)


def clean_dataset(dataset: pd.DataFrame):
    """
    处理数据，每一列用均值填充缺失值，然后标准化 N(0,1)
    fit_transform = fit + transform
    """
    if not dataset.empty:
        impute = SimpleImputer(missing_values=np.nan, strategy="mean")
        temp = impute.fit_transform(dataset.iloc[:, 1:-1])
        scaler = StandardScaler()
        temp = scaler.fit_transform(temp)

        apkName = dataset.iloc[:, 0]
        clas = dataset.iloc[:, -1]

        dataset_temp = combine_dataset(temp, clas, apkName, dataset.columns)
        return dataset_temp
    return pd.DataFrame()


def split_dataset(dataset: pd.DataFrame):
    """
    cols for dataset: ['apkName', ..., 'label']
    dataset.shape = (rows, cols)

    Split the dataset into X, y, and apkName,
    where y is label column, X is the rest of the columns, and apkName is the first column.
    """
    X = dataset.values[:, 1:-1]
    y = dataset.values[:, -1]
    apkName = dataset.values[:, 0]
    return [X, y, apkName]


def delete_feature_with_zero_variance(dataset):
    """
    移除方差为 0 的特征列。保持列名一致。
    """
    [X, y, apkName] = split_dataset(dataset)
    vt = VarianceThreshold()  # (.8 * (1 - .8))
    X_vt = vt.fit_transform(X)
    idx = [0]
    # idx.append(0)
    for i in range(0, len(vt.get_support(indices=True))):
        idx.append(vt.get_support(indices=True)[i] + 1)
    idx.append(X.shape[-1] + 1)
    return combine_dataset(X_vt, y, apkName, dataset.columns[idx])


def remove_instances(dataset: pd.DataFrame, number):
    """
    在消除 0 方差特征后，删除标签中 < number 个的实例。（再次检验 apk per author）
    """
    vc = dataset.iloc[:, -1].value_counts()
    u = [i not in set(vc[vc < number].index) for i in dataset.iloc[:, -1]]
    dataset = dataset[u]
    return dataset


def feature_selection(dataset, percentile):
    """
    从数据集中选择前 percentile 百分比的最重要特征。
    """
    selector = SelectPercentile(score_func=f_classif, percentile=percentile)
    [X, y, z] = split_dataset(dataset)
    X_new = selector.fit_transform(X, y)
    if X_new.size == 0:
        return pd.DataFrame()

    idx = [0]
    # idx.append(0)
    for i in range(0, len(selector.get_support(indices=True))):
        idx.append(selector.get_support(indices=True)[i] + 1)
    idx.append(X.shape[-1] + 1)

    return combine_dataset(X_new, y, z, dataset.columns[idx])


def crossval_TFPN(
    dataset: pd.DataFrame,
    cv: int,
    epoch: int,
    tfpn_counter: TFPNCounter,
    original_dataset: pd.DataFrame,
    owriter: AssociationOutputWriter,
):
    """
    Modified cross validation function to calculate TP, TN, FP, FN.

    The original cross validation function is deleted, check the git history for the original code.
    """
    # fmt: off
    random_states = [12, 22, 32, 42, 52, 358, 52, 229, 879, 272, 330, 263, 530, 614, 456, 943, 224, 559, 349, 612, 407, 60, 639, 354, 662, 767, 246, 852, 816, 502, 896, 968, 829, 128, 168, 203, 740,643, 445, 326, 560, 154, 693, 500, 190, 949, 625, 239, 391, 693]
    # fmt: on

    X = dataset.values[:, 1:-1]
    y = dataset.values[:, -1]

    classifiers: List[Tuple[str, GaussianNB]] = [
        # ("GNB", GaussianNB()),
        # ("KNN", KNeighborsClassifier(n_jobs=40)),
        # ("SVC", SVC()),
        ("RF", RandomForestClassifier(random_state=42, n_jobs=40)),
        # (
        #     "LGBM",
        #     lgb.LGBMClassifier(
        #         random_state=42, boosting_type="gbdt", n_jobs=40, learning_rate=0.05
        #     ),
        # ),
    ]

    for name, model in classifiers:
        kfold = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_states[0]
        )
        for train_idx, test_idx in kfold.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            # y_pred = model.predict(original_dataset.values[:, 1:-1])
            y_pred = model.predict(X_test)

            TP, FP, TN, FN = tfpn_counter.count_TP_FP_TN_FN(
                # original_dataset.values[:, -1], y_pred
                y_test,
                y_pred,
            )
            print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
            owriter.dump_trainset(dataset.iloc[train_idx, :])
            # owriter.dump_pred(original_dataset, y_pred)
            owriter.dump_pred(dataset.iloc[test_idx, :], y_pred)
            owriter.dump_testset(dataset.iloc[test_idx, :])


if __name__ == "__main__":

    db_name = sys.argv[1]  #'malware'
    rq = db_name + ".csv"
    epoch = 5
    cv = 2
    apk_number_per_author = 1
    percentiles = [100]
    feature_type = sys.argv[2]  #'customsmali_perm_lib'
    number_of_ngram = 10000
    ngram_type = "all"
    tfpn_counter = TFPNCounter(
        Path(f"./apk/{db_name}"), lambda author, apks: len(apks) <= 1
    )
    owriter = AssociationOutputWriter(
        Path(f"./result/{db_name}-50-50"), tfpn_counter.author_apks
    )

    result_folder = os.getcwd() + "/result/"
    feature_folder = os.getcwd() + "/feature/"

    f_file = "{}/{}_{}.csv".format(result_folder, db_name.split("/")[-1], rq)
    f_features_file = "{}/{}_{}_feature.csv".format(
        result_folder, db_name.split("/")[-1], rq
    )
    with open(f_file, "a") as f:
        with open(f_features_file, "a") as f_features:
            if feature_type == "ngram":
                feature_type += "/" + ngram_type
            if "ngram" in feature_type:
                type = "{}_{}".format(feature_type.split("/")[-1], number_of_ngram)
            elif "metadata" in feature_type:
                type = "{}_{}".format(feature_type.split("/")[-1], 50000)
            else:
                type = feature_type

            dataset = read_all_csv_from_db_folder(
                "{}{}/{}".format(feature_folder, db_name, feature_type),
                type,
                number_of_ngram,
            )
            dataset = clean_dataset(dataset)
            dataset = delete_feature_with_zero_variance(dataset)
            dataset_ = dataset.copy(deep=True)
            dataset = remove_instances(dataset, apk_number_per_author)
            for perc in percentiles:
                if not dataset.empty:
                    dataset1 = feature_selection(dataset, perc)
                crossval_TFPN(dataset1, cv, epoch, tfpn_counter, dataset_, owriter)
