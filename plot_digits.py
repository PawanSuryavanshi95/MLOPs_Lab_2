# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import numpy as np

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

hyper_comb = [{ 'gamma': gamma, 'C': c } for gamma in gamma_list for c in c_list]

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

digits = datasets.load_digits()

images = digits.images

print("Images are reshaped to 5 x 5")

resized_images = []

for image in digits.images:
    image_resized = resize(image, (5, 5), anti_aliasing=True)
    resized_images.append(image_resized)

resized_images = np.asarray(resized_images)

# first is the number of rows in the image, second is the number of items in each row
img_size = (len(resized_images[0]), len(resized_images[0][0]))

print("Number of Rows in each image : ", img_size[0])
print("Number of item in each row : ", img_size[1], "\n")
print("Size of the image : ", img_size, "\n")

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, resized_images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# flatten the images
n_samples = len(digits.images)
data = resized_images.reshape((n_samples, -1))

# Split data into 80% train and 20% test + dev subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size = 1 - train_frac, shuffle=True
)

# Split data into 50% test and 50% dev subsets
X_test, X_dev, y_test, y_dev = train_test_split(
    X_test, y_test, test_size = dev_frac / (test_frac + dev_frac) , shuffle=True
)

best_model = None
best_accuracy = -1
best_hyper_params = None

d = {1: ["Python", 33.2, 'UP'],
2: ["Java", 23.54, 'DOWN'],
3: ["Ruby", 17.22, 'UP'],
10: ["Lua", 10.55, 'DOWN'],
5: ["Groovy", 9.22, 'DOWN'],
6: ["C", 1.55, 'UP']
}
print ("{:<27} {:<27} {:<27} {:<27}".format('Hyperparameters','Training Accuracy','Validation Accuracy','Testing Accuracy'))
print("")
for comb in hyper_comb:

    # Create a classifier: a support vector classifier
    clf = svm.SVC()

    clf.set_params(**comb)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_dev)
    validation_accuracy = metrics.accuracy_score(y_pred = predicted, y_true = y_dev)

    predicted = clf.predict(X_test)
    testing_accuracy = metrics.accuracy_score(y_pred = predicted, y_true = y_test)

    predicted = clf.predict(X_train)
    training_accuracy = metrics.accuracy_score(y_pred = predicted, y_true = y_train)

    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        best_model = clf
        best_hyper_params = comb

    print ("{:<27} {:<27} {:<27} {:<27}".format(str("gamma = " + str(comb["gamma"]) + ", C = " + str(comb["C"])), training_accuracy, validation_accuracy, testing_accuracy ))

    # _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    # for ax, image, prediction in zip(axes, X_test, predicted):
    #     ax.set_axis_off()
    #     image = image.reshape(8, 8)
    #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    #     ax.set_title(f"Prediction: {prediction}")


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