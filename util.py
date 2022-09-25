import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import resize

def preprocess_data(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    return (data, dataset.target)

def data_viz(dataset):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, dataset.images, dataset.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

def train_test_split_2(X, y, train_frac, test_frac, dev_frac):
    # Split data into 80% train and 20% test + dev subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 1 - train_frac, shuffle=True
    )

    # Split data into 50% test and 50% dev subsets
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_test, y_test, test_size = dev_frac / (test_frac + dev_frac) , shuffle=True
    )

    return (X_train, y_train, X_dev, y_dev, X_test, y_test)

def hyperparameter_tuning(X_train, y_train, X_test, y_test, X_dev, y_dev, hyper_comb, metric):

    best_model = None
    best_accuracy = -1
    best_hyper_params = None

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
        validation_accuracy = metric(y_pred = predicted, y_true = y_dev)

        predicted = clf.predict(X_test)
        testing_accuracy = metric(y_pred = predicted, y_true = y_test)

        predicted = clf.predict(X_train)
        training_accuracy = metric(y_pred = predicted, y_true = y_train)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_model = clf
            best_hyper_params = comb

        print ("{:<27} {:<27} {:<27} {:<27}".format(str("gamma = " + str(comb["gamma"]) + ", C = " + str(comb["C"])), training_accuracy, validation_accuracy, testing_accuracy ))
    
    return (best_model, best_accuracy, best_hyper_params)