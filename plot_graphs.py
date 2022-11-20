# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
import pdb
from sklearn.metrics import f1_score, accuracy_score

from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb,
    tune_and_save,
)
from joblib import dump, load

import argparse, os

parser = argparse.ArgumentParser(description ='Enter classifier name and random seed value')
  
# Adding Arguments
parser.add_argument('--clf_name')
parser.add_argument('--random_state')

args = parser.parse_args()
print(args.clf_name, args.random_state)

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

params = {}
params["gamma"] = gamma_list
params["C"] = c_list

h_param_comb = get_all_h_param_comb(params)


# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits

seed = int(args.random_state)

x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac, seed
)

# PART: Define the model
# Create a classifier: a support vector classifier

clf = None

if(args.clf_name == 'svm'):
    clf = svm.SVC()
elif(args.clf_name == 'tree'):
    clf = DecisionTreeClassifier()

# define the evaluation metric
metric = metrics.accuracy_score

actual_model_path = tune_and_save(
    clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, seed, model_path=None
)

# 2. load the best_model
best_model = load(actual_model_path)

# PART: Get test set predictions
# Predict the value of the digit on the test subset
predicted = best_model.predict(x_test)

report = metrics.classification_report(y_test, predicted)

test_acc = accuracy_score(y_test, predicted)
test_f1 = f1_score(y_test, predicted, average='macro')

pred_image_viz(x_test, predicted)

model_type = actual_model_path.split("_")[0].split('/')[1]

txt_file_name = model_type + '_' + str(seed) + '.txt'

print(model_type, txt_file_name)

txt_content = f"""test accuracy: {test_acc}
test macro-f1: {test_f1} 
model saved at ./{actual_model_path}"""

if not os.path.exists('results'):
    os.mkdir('results')

with open('results/' + txt_file_name, 'w') as f:
    f.write(txt_content)

# 4. report the test set accurancy with that best model.
# PART: Compute evaluation metrics
print(
    f"Classification report for classifier {clf}:\n"
    f"{report}\n"
)
