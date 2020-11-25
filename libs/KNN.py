import os 
import pandas as pd
import numpy as np 
import json
import itertools
from fpdf import FPDF

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from yellowbrick.classifier import ClassificationReport

import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from libs.utils import create_folder

class KNN(object):
    """This class performs KNN algorithm
    """

    def __init__(self, data, weights='uniform', n_neighbors=15):
        """Initialize the class
        
        # Arguments
            - data: a list containing data_train, data_test, data_val, label_train, label_test, label_value
            - weights: weight function used in prediction; must be 'uniform' or 'distance'
            - n_neighbors: number of nearest neighbor consider for the classifier
        """

        # Load JSON file
        with open('config/config.json', 'r') as f:
            self.cfg = json.load(f)

        # Create object of the Classifier
        self.n_neighbors = n_neighbors
        self.neigh = KNeighborsClassifier(weights=weights, n_neighbors=self.n_neighbors)

        # Read data
        self.data_train = data[0]
        self.label_train = data[1]
        self.data_test = data[2] 
        self.label_test = data[3]
                
        # Write info data
        create_folder(self.cfg['info_path'])
        create_folder(self.cfg['plot_path'])

    def train(self):
        """ Train the model 
        """

        # Train the algorithm on training data and predict using the validation data
        self.model = self.neigh.fit(self.data_train, self.label_train)
        self.pred = self.model.predict(self.data_test)

        # Evaluate accuracy
        self.acc = accuracy_score(self.label_test, self.pred)
        print (f"[ACCURACY] : {self.acc}")


    def plot_precision_recall_f1(self, classes=['Won','Loss'], display=False):
        """ Plot Precision Recall F1

        # Arguments:
            - classes: A list of all labels
            - display: boolean value for showing plot or not; default is False
        """

        self.train()
       
        # Instantiate the classification model and visualizer
        visualizer = ClassificationReport(self.neigh, classes=classes)
        visualizer.fit(self.data_train, self.label_train) # Fit the training data to the visualizer
        visualizer.score(self.data_test, self.label_test) # Evaluate the model on the test data
        visualizer.poof(outpath=self.cfg['plot_path'] + "knn-report.png") # save the data
        if display:
            g = visualizer.poof() # Draw/show/poof the data


    def confusion_matrix(self, classes=['Won','Loss'], display=False):
        """ Plot the Confusion Matrix
        
        # Arguments:
            - classes: A list of all labels
            - display: boolean value for showing plot or not; default is False

        """
        self.classes = classes
        self.train()
        
        cf = confusion_matrix(self.label_test, self.pred)
        lpp = cf[0,1] + cf[1,1]
        lap = cf[1,0] + cf[1,1]
        self.lp = cf[1,1]/lpp
        self.lr = cf[1,1]/lap
        self.lf1 = 2 * (self.lp * self.lr) / (self.lp + self.lr)
        print(f'{self.classes[1]} precision: {self.lp}, recall: {self.lr}, f1 {self.lf1}')

        wpp = cf[0,0] + cf[1,0]
        wap = cf[0,0] + cf[0,1]
        self.wp = cf[0,0]/wpp
        self.wr = cf[0,0]/wap
        self.wf1 = 2 * (self.wp * self.wr) / (self.wp + self.wr)
        print(f'{self.classes[0]} precision: {self.wp}, recall: {self.wr}, f1 {self.wf1}')

        # Plot non-normalized confusion matrix
        titles_options = [("Confusion matrix, without normalization", None),
                        ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(self.model, self.data_test, self.label_test,
                                        display_labels=classes,
                                        cmap=plt.cm.Blues,
                                        normalize=normalize)
            disp.ax_.set_title(title)
            text = "-normalize" if normalize else ""
            plt.savefig(self.cfg['plot_path'] + f"knn-confusion-matrix{text}.png")

        print(f"[CONFUSION MATRIX]:\n{cf}\n")

        # Display plot 
        plt.show() if display else plt.close('all')


    def plot_decision_boundary_2D(self, 
                            csv_path="/../output/dataset.csv", 
                            cols=['RMs', 'LMs', 'UMs'],
                            labels=['Class'], 
                            weights='uniform',
                            display=False):
        """ Plot the Decision Boundary in 2D (2-Class Classifier)
        
        # Arguments:
            - csv_path: the folder/file path of the csv to read in. If folder dataframes are appended
            - cols: list of features
            - labels: list of labels
            - weights: weight function used in prediction; must be 'uniform' or 'distance'
            - n_neighbors: number of nearest neighbor consider for the classifier
            - weights: value for the weights used in the algorithm must be 'uniform' or 'distance'
            - display: boolean value for showing plot or not; default is False
        """

        # Load csv file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + csv_path

        df = pd.read_table(path, sep=',')
        x = df[cols]
        y = df[labels]

        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        cmap_bold  = ListedColormap(['#FF0000', '#00FF00'])

        mesh_step_size = .02  # step size in the mesh
        plot_symbol_size = 50
        
        for plane in itertools.combinations([0,1,2], 2):
            
            x_mat = x.values[:, plane]
            y_mat = y.values[:, :2]

            axes = ''
            for i in plane:
                axes +=  ''.join(c for c in ['x', 'y', 'z'][i])

            # fit the data using KNN Classifier
            self.neigh.fit(x_mat, y_mat)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].

            x_min, x_max = x_mat[:, 0].min() - 1, x_mat[:, 0].max() + 1
            y_min, y_max = x_mat[:, 1].min() - 1, x_mat[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                                np.arange(y_min, y_max, mesh_step_size))
            Z = self.neigh.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure()
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

            # Plot training points
            plt.scatter(x_mat[:, 0], x_mat[:, 1], s=plot_symbol_size, c=y_mat, cmap=cmap_bold, edgecolor = 'black')
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())

            patch0 = mpatches.Patch(color='#FF0000', label='danger')
            patch1 = mpatches.Patch(color='#00FF00', label='normal')
            plt.legend(handles=[patch0, patch1])

            plt.xlabel(f'{cols[plane[0]]}')
            plt.ylabel(f'{cols[plane[1]]}')

            plt.title(f"2-Class classification (plane:{axes} k = {self.n_neighbors}, weights = {weights})")
            plt.savefig(self.cfg['plot_path'] + f"knn-decision-boundary-{axes}.png")
        
        # Display plot 
        plt.show() if display else plt.close('all')

    def plot_decision_boundary_3D(self, 
                            csv_path="/../output/dataset.csv", 
                            cols=['RMs', 'LMs', 'UMs'],
                            labels=['Class'],
                            weights='uniform', 
                            display=False):
        """ Plot the Decision Boundary in 3D (2-Class Classifier)
        
        # Arguments:
            - csv_path: the folder/file path of the csv to read in. If folder dataframes are appended
            - cols: list of features
            - labels: list of labels
            - weights: weight function used in prediction; must be 'uniform' or 'distance'
            - n_neighbors: number of nearest neighbor consider for the classifier
            - display: boolean value for showing plot or not; default is False
        """

        # Load csv file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + csv_path

        df = pd.read_table(path, sep=',')
        x = df[cols]
        y = df[labels]

        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        cmap_bold  = ListedColormap(['#FF0000', '#00FF00'])

        # step size in the mesh
        mesh_step_size = .02 # np.arange
        space_sampling_points = 20 # np.linspace
        plot_symbol_size = 20

        x_mat = x.values[:, :3]
        y_mat = y.values[:, :3]

        # fit the data using KNN Classifier
        self.neigh.fit(x_mat, y_mat)

        # Draw 3D Plot
        x_min, x_max = x_mat[:, 0].min() - 1, x_mat[:, 0].max() + 1
        y_min, y_max = x_mat[:, 1].min() - 1, x_mat[:, 1].max() + 1
        z_min, y_max = x_mat[:, 2].min() - 1, x_mat[:, 2].max() + 1
        xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, space_sampling_points),
                            np.linspace(y_min, y_max, space_sampling_points), 
                            np.linspace(z_min, y_max, space_sampling_points))
        
        Z = self.neigh.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])[: ,-1]
        Z = Z.reshape(xx.shape)

        # Put the result into a color plot
        fig = plt.figure()            
        ax = fig.gca(projection='3d')

        # Plot the separating hyperplane by recreating the isosurface for the distance == 0 
        # level in the distance grid computed through the decision function 
        verts, faces, _, _ = measure.marching_cubes(Z, 0)
        # Scale and transform to actual size of the interesting volume
        verts = verts * [x_max - x_min, y_max - y_min, y_max - z_min] / space_sampling_points
        verts = verts + [x_min, y_min, z_min]
        # and create a mesh to display
        mesh = Poly3DCollection(verts[faces], facecolor='green', edgecolor='green', alpha=0.5)
        ax.add_collection3d(mesh)

        # Plot training points
        ax.scatter3D(x_mat[:, 0], x_mat[:, 1], x_mat[:, 2], s=plot_symbol_size, c=y_mat, cmap=cmap_bold, edgecolor = 'black')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_zlim(zz.min(), zz.max())

        patch0 = mpatches.Patch(color='#FF0000', label='danger')
        patch1 = mpatches.Patch(color='#00FF00', label='normal')
        fig.legend(handles=[patch0, patch1])

        ax.set_xlabel(f'{cols[0]}')
        ax.set_ylabel(f'{cols[1]}')
        ax.set_zlabel(f'{cols[2]}')

        for plane in itertools.combinations([0,1,2], 2):
        
            x_mat = x.values[:, plane]
            y_mat = y.values[:, :2]

            axes = ''
            for i in plane:
                axes +=  ''.join(c for c in ['x', 'y', 'z'][i])

            # fit the data using KNN Classifier
            self.neigh.fit(x_mat, y_mat)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].

            x_min, x_max = x_mat[:, 0].min() - 1, x_mat[:, 0].max() + 1
            y_min, y_max = x_mat[:, 1].min() - 1, x_mat[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                                np.arange(y_min, y_max, mesh_step_size))
            Z = self.neigh.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            #plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

            empty_ax = ''.join(i for i in (set(['x', 'y', 'z']) - set(list([c for c in axes]))))

            if empty_ax == 'z':
                offset = 0
            elif empty_ax == 'y':
                offset = 3.5
            elif empty_ax == 'x':
                offset = -0.5

            ax.contourf(xx, yy, Z, zdir='y', offset=3.5, cmap=cmap_light)
            ax.contourf(xx, yy, Z, zdir='x', offset=-0.5, cmap=cmap_light)
            ax.contourf(xx, yy, Z, zdir='z', offset=0, cmap=cmap_light)

        plt.title(f"2-Class classification 3D (k = {self.n_neighbors}, weights = {weights})")
        plt.savefig(self.cfg['plot_path'] + f"knn-decision-boundary-{weights}-3D.png")
        
        # Display plot 
        plt.show() if display else plt.close('all')

    def generate_report(self):
        """ Generate a report as PDF file
        """
        # Option
        pdf = FPDF()
        pdf.add_page()
        pdf.set_xy(20, 5)
        pdf.set_title("KNN")
        pdf.accept_page_break()

        # Title 
        pdf.set_font('arial', 'B', 25)
        pdf.cell(0, 30, 'KNN', border=0, ln=2, align='C')

        # Info 
        pdf.cell(15)
        pdf.set_font('arial', 'I', 14)
        pdf.cell(30, 2, 'Accuracy : ', border=0, ln=0, align='L')
        pdf.set_font('arial', '', 14)
        pdf.cell(50, 2, f'{self.acc}', border=0, ln=2, align='L')
        
        # Plot Report
        pdf.cell(10)
        pdf.image(self.cfg['plot_path'] + "knn-report.png", x = 30, y = 40, w = 150, h = 100, type = '', link = '')

        # Chapter 
        pdf.set_y(140)
        pdf.cell(10)
        pdf.set_font('arial', 'B', 18)
        pdf.cell(0, 20, 'Confusion Matrix', border=0, ln=2, align='L')

        # Plot Confusion Matrix
        pdf.cell(0)
        pdf.image(self.cfg['plot_path'] + "knn-confusion-matrix.png", x = 20, y = 160, w = 100, h = 75, type = '', link = '')  
        pdf.cell(10)
        pdf.image(self.cfg['plot_path'] + "knn-confusion-matrix-normalize.png", x = 100 , y = 160, w = 100, h = 75, type = '', link = '')

        # Info -> Precision/Recall/F1
        pdf.set_y(235)
        pdf.cell(15)
        pdf.set_font('arial', 'B', 12)
        pdf.cell(30, 7, f'{self.classes[0]} : ', border=0, ln=0, align='L')
        pdf.cell(50)
        pdf.cell(30, 7, f'{self.classes[1]} : ', border=0, ln=1, align='L')

        pdf.cell(20)
        pdf.set_font('arial', 'I', 12)
        pdf.cell(22, 7, f'Precision : ', border=0, ln=0, align='L')
        pdf.set_font('arial', '', 12)
        pdf.cell(60, 7, f'{self.lp}', border=0, ln=0, align='L')
        pdf.set_font('arial', 'I', 12)
        pdf.cell(22, 7, f'Precision : ', border=0, ln=0, align='L')
        pdf.set_font('arial', '', 12)
        pdf.cell(60, 7, f'{self.wp}', border=0, ln=1, align='L')

        pdf.cell(20)
        pdf.set_font('arial', 'I', 12)
        pdf.cell(22, 7, f'Recall : ', border=0, ln=0, align='L')
        pdf.set_font('arial', '', 12)
        pdf.cell(60, 7, f'{self.lr}', border=0, ln=0, align='L')
        pdf.set_font('arial', 'I', 12)
        pdf.cell(22, 7, f'Recall : ', border=0, ln=0, align='L')
        pdf.set_font('arial', '', 12)
        pdf.cell(60, 7, f'{self.wr}', border=0, ln=1, align='L')

        pdf.cell(20)
        pdf.set_font('arial', 'I', 12)
        pdf.cell(22, 7, f'F1 : ', border=0, ln=0, align='L')
        pdf.set_font('arial', '', 12)
        pdf.cell(60, 7, f'{self.lf1}', border=0, ln=0, align='L')
        pdf.set_font('arial', 'I', 12)
        pdf.cell(22, 7, f'F1 : ', border=0, ln=0, align='L')
        pdf.set_font('arial', '', 12)        
        pdf.cell(60, 7, f'{self.wf1}', border=0, ln=1, align='L')

        pdf.add_page()
        pdf.set_font('arial', 'B', 18)
        pdf.cell(10)
        pdf.cell(0, 20, 'Decision Boundary', border=0, ln=2, align='L')

        try:
            pdf.image(self.cfg['plot_path'] + "knn-decision-boundary-xy.png", x = 10, y = 30, w = 100, h = 70, type = '', link = '')
            pdf.image(self.cfg['plot_path'] + "knn-decision-boundary-xz.png", x = 100, y = 30, w = 100, h = 70, type = '', link = '')
            pdf.image(self.cfg['plot_path'] + "knn-decision-boundary-yz.png", x = 10, y = 100, w = 100, h = 70, type = '', link = '')
        except:
            pass
        try:
            pdf.image(self.cfg['plot_path'] + "knn-decision-boundary-uniform-3D.png", x = 10, y = 175, w = 180, h = 120, type = '', link = '')
        except:
            print("ciao")
        
        pdf.output(self.cfg['info_path'] + 'knn-report.pdf', 'F')