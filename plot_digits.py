# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

from util import *

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

hyper_comb = [{ 'gamma': gamma, 'C': c } for gamma in gamma_list for c in c_list]

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

digits = datasets.load_digits()

# images = digits.images

# print("Images are reshaped to 5 x 5")

# resized_images = []

# for image in digits.images:
#     image_resized = resize(image, (5, 5), anti_aliasing=True)
#     resized_images.append(image_resized)

# resized_images = np.asarray(resized_images)

# # first is the number of rows in the image, second is the number of items in each row
# img_size = (len(resized_images[0]), len(resized_images[0][0]))

# print("Number of Rows in each image : ", img_size[0])
# print("Number of item in each row : ", img_size[1], "\n")
# print("Size of the image : ", img_size, "\n")

data_viz(digits)

(data, target) = preprocess_data(digits)

(X_train, y_train, X_dev, y_dev, X_test, y_test) = train_test_split_2(data, target, train_frac, test_frac, dev_frac)

metric = metrics.accuracy_score
(best_model, best_accuracy, best_hyper_params) = hyperparameter_tuning(X_train, y_train, X_test, y_test, X_dev, y_dev, hyper_comb, metric)

predicted = best_model.predict(X_test)

print("\nBest Hyperparameters obtained : ", best_hyper_params)

predicted = best_model.predict(X_train)
training_accuracy = metrics.accuracy_score(y_pred = predicted, y_true = y_train)

predicted = best_model.predict(X_dev)
validation_accuracy = metrics.accuracy_score(y_pred = predicted, y_true = y_dev)

predicted = best_model.predict(X_test)
testing_accuracy = metrics.accuracy_score(y_pred = predicted, y_true = y_test)

print("\nTraining Accuracy on Best Model : ", training_accuracy)
print("Validation Accuracy on Best Model : ", validation_accuracy)
print("Testing Accuracy on Best Model : ", testing_accuracy)