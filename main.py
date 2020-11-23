import json 

from libs.NaiveBayes import NaiveBayes
from libs.LinearSVC import Linear_SVC
from libs.KNN import KNN
#from libs.DecisionTreeM import DecisionTreeM
from libs.LogisticRegression import Logistic_Regression
from libs.LinearDiscriminantAnalysis import LDA
from libs.SupportVectorMachine import SVM
from libs.utils import split_data

if __name__ == '__main__':
    
    # Load JSON file
    with open('config/config.json', 'r') as f:
        cfg = json.load(f)

    # Change the utils preprocess function if you want to change classes
    classes = cfg['classes']

    # csv path can be folder or single file
    data = split_data(cfg['csv_path'])
    x_train, y_train, x_test, y_test = data[0:4]

    # Naive Bayes
    if "Naive-Bayes" in cfg['classifiers']:
        nb = NaiveBayes(data)
        nb.confusion_matrix(classes)
        nb.plot_precision_recall_f1(classes)
        nb.plot_decision_boundary(cols=cfg['csv_cols'], labels=cfg['csv_labels'])
        nb.generate_report()

    # Linear SVC
    if "LinearSVC" in cfg['classifiers']:
        lsvc = Linear_SVC(data)
        lsvc.confusion_matrix(classes)
        lsvc.plot_precision_recall_f1(classes)
        lsvc.plot_decision_boundary(cols=cfg['csv_cols'], labels=cfg['csv_labels'])
        lsvc.generate_report()

    # KNN
    if "KNN" in cfg['classifiers']:
        knn = KNN(data)
        knn.confusion_matrix(classes)
        knn.plot_precision_recall_f1(classes)
        knn.plot_decision_boundary(cols=cfg['csv_cols'], labels=cfg['csv_labels'])
        knn.generate_report()

    # Decision Tree
    # if "DecisionTree" in cfg['classifiers']:
    #     dtree = DecisionTreeM(data)
    #     dtree.train()

    # Logistic Regression
    if "LogisticRegression" in cfg['classifiers']:
        lr = Logistic_Regression(data)
        lr.confusion_matrix(classes)
        lr.plot_precision_recall_f1(classes)
        lr.plot_decision_boundary(cols=cfg['csv_cols'], labels=cfg['csv_labels'])
        lr.generate_report()

    # Linear Discriminant Analysis
    if "LDA" in cfg['classifiers']:
        lda = LDA(data)
        lda.confusion_matrix(classes)
        lda.plot_precision_recall_f1(classes)
        lda.plot_decision_boundary(cols=cfg['csv_cols'], labels=cfg['csv_labels'])
        lda.generate_report()

    # Support Vector Machine
    if "SVM" in cfg['classifiers']:
        svm = SVM(data)
        svm.confusion_matrix(classes)
        svm.plot_precision_recall_f1(classes)
        svm.plot_decision_boundary(cols=cfg['csv_cols'], labels=cfg['csv_labels'])
        svm.generate_report()