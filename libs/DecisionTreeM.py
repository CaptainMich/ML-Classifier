import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 

from sklearn.tree import export_graphviz

# Module sklearn.externals.six was removed in version 0.23. 
# if you want to use this module, you have to downgrade to version 0.22 or lower.
from sklearn.externals.six import StringIO  

from IPython.display import Image  
import pydotplus

class DecisionTreeM(object):
    """This class performs Decision Tree algorithm
    """

    def __init__(self, data):
        """Initialize the class
        
        # Arguments
            - data: a list containing data_train, data_test, data_val, label_train, label_test, label_value
        """
        # Create object of the Classifier
        self.dectree = DecisionTreeClassifier()

        # Read data
        self.data_train = data[0]
        self.label_train = data[1]
        self.data_test = data[2] 
        self.label_test = data[3]
        

    def train(self):
        """ Train the model 
        """
        # Train the algorithm on training data and predict using the validation data
        self.model = self.dectree.fit(self.data_train,self.label_train)
        self.y_pred = self.model.predict(self.data_test)
 
        # Model Accuracy, how often is the classifier correct?
        print(f"[ACCURACY]: {metrics.accuracy_score(self.label_test, self.y_pred)}")
 
    def print_graph(self, 
                    feature_cols = [ 'Ms1','Ms2','Ms3', 'Ms4','Ms5'],
                    classnames = ['Win','Loss'],
                    output= './output/'):
        """Plot Tree Graph

        # Arguments:
            - feature_cols: list of dataset features and target variable
            - classname: a list containing all the class names
            - output: the output path
        """
        self.feature_cols = feature_cols
        self.classnames = classnames

        dot_data = StringIO()
        export_graphviz(self.dectree, out_file=dot_data,  
                        filled=True, rounded=True,
                        special_characters=True,feature_names=self.feature_cols,class_names=self.classnames)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
        graph.write_png(output + 'Tree.png')
        Image(graph.create_png())
 