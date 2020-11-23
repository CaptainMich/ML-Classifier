# ML-Classifier

This project aims to identify ***"normal"*** or ***"danger"*** objects in conveyor belt of an airport 

### Dataset

Dataset is made by the measures of three different sensors, ***'RMs'*** ***'LMs'*** ***'UMs'*** placed respectively in the right, in the left and up. <br> 
Format is the following 

| RMs | LMs | UMs | Class |
| --- | --- | --- |  --- |
|1.98242 | 2.017493 | 2.015046 | Object

### Configuration

Configuration file entries are:

- ***csv_path*** : the path of csv dataset folder; can be also one single file
- ***classes*** : list of the two classes identifier since the classifier is binary
- ***classifiers*** : list containing all the classifiers we want to use. All possibilities are: <br>
                 *["Naive-Bayes", "LinearSVC", "KNN", "DecisionTree", "LogisticRegression", "LDA", "SVM"]*
- ***csv_cols*** : list of features; must contains only 2 values
- ***csv_labels*** : list of labels
- ***normal*** : list of normal objects 
- ***danger*** : list of danger objects
- ***plot_path*** : the path of the plot output
- ***info_path*** : the path of the pdf output

### Requirements 

- pandas
- fpdf
- sklearn
- matplotlib
- yellowbrick

### How it works?

Run `main.py` and see a PDF report in the *./output/info* (one for each selected classifier is generated) and all the plot in the *./output/plot*
