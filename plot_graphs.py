# Import datasets, classifiers and performance metrics
from unittest import result
from sklearn import datasets, svm, metrics, tree
import pdb
import numpy as np

from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb,
    tune_and_save,
    macro_f1,
)
from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# SVC Hyperparameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

svc_params = { "gamma": gamma_list, "C": c_list }

# Decision Tree Hyperparameters
max_depth_list = [2, 10, 20, 50, 100]
min_samples_split_list = [2, 3, 5]

dtc_params = { "max_depth": max_depth_list, "min_samples_split": min_samples_split_list }

svc_comb = get_all_h_param_comb(svc_params)
dtc_comb = get_all_h_param_comb(dtc_params)

h_param_comb = {"svc":svc_comb, "dtc":dtc_comb}

# loading dataset
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)

del digits

n_cv = 5
results = {}

svc = svm.SVC()
dtc = tree.DecisionTreeClassifier()

model_types = {
    "svc": svc,
    "dtc": dtc,
}

for model in model_types:
    results[model] = []

metric_list = [metrics.accuracy_score, macro_f1]
h_metric = metrics.accuracy_score

for i in range(n_cv):

    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac
    )

    for model in model_types:
        clf = model_types[model]

        actual_model_path = tune_and_save(
            clf, x_train, y_train, x_dev, y_dev, h_metric, h_param_comb[model], model_path=None
        )

        best_model = load(actual_model_path)

        predicted = best_model.predict(x_test)

        results[model].append({m.__name__:m(y_pred=predicted, y_true=y_test) for m in metric_list})
        
        # 4. report the test set accurancy with that best model.
        print(
            f"Classification report for classifier {clf}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
        )

print("\nResults :-")
for model in model_types:
    print("\nAccuracy for {} :".format(model))
    L = []
    for metric_result in results[model]:
        L.append(metric_result['accuracy_score'])
        print(metric_result['accuracy_score'])
    L = np.asarray(L)
    print("\n Mean : {} \n Standard Deviation : {}".format(np.mean(L), np.std(L)))