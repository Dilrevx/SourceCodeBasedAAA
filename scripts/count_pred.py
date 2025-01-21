import json
from pathlib import Path

import numpy as np

PRED_PATH = Path("./result/sdkid-creator-path-50-50/pred.json")


def main():
    print("Loading json file...")
    json_ = json.load(PRED_PATH.open())
    print(json_)

    TP, FP, TN, FN = 0, 0, 0, 0
    for author in json_:
        groud_truth = np.array(json_[author]["ground_truth"])
        predicted = np.array(json_[author].get("predicted", []))

        if groud_truth.shape == predicted.shape and (groud_truth == predicted).all():
            TP += np.sum(groud_truth == predicted)
        else:
            raise NotImplementedError("I don't know how to calculate FP, TN, FN")
        """
            for author in total_res:
                if "predicted" not in total_res[author]:
                    print(author + ": " + str(total_res[author]))
                    fn += len(total_res[author]["ground_truth"])
                    continue
                if "ground_truth" not in total_res[author]:
                    print(author + ":" + str(total_res[author]))
                    fp += len(total_res[author]["predicted"])
                    continue
                for apk in total_res[author]["predicted"]:
                    if apk in total_res[author]["ground_truth"]:
                        tp += 1
                for apk in total_res[author]["predicted"]:
                    if apk not in total_res[author]["ground_truth"]:
                        fp += 1
                for apk in total_res[author]["ground_truth"]:
                    if apk not in total_res[author]["predicted"]:
                        fn += 1
        """


main()
