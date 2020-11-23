import os 
import json
import pandas as pd
import shutil
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def preprocess(csv_path, output='./output', display=False):
    """This function read the csv data. It also transform features to number if any 

    # Arguments
        - csv_path: the folder/file path of the csv to read in. If folder dataframes are appended
        - display: boolean for print more info 
    # Return
        - df: the csv in panda table format (dataframe)
    """

    # Load JSON file
    with open('config/config.json', 'r') as f:
        cfg = json.load(f)

    normal = cfg['normal']
    danger = cfg['danger']

    # Create the Labelencoder object
    le = preprocessing.LabelEncoder()

    # Load csv file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + csv_path
    
    appended_data = []
    # check if input is a folder
    if (os.path.isdir(path)):
        for _dirname, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                csv_path = path + filename
                data = pd.read_csv(csv_path, sep=',')

                for c in data['Class']:
                    if c in normal:
                        data['Class'] = 'Normal'
                    elif c in danger:
                        data['Class'] = 'Danger'

                appended_data.append(data)

        df = pd.concat(appended_data)

    else:
        # Read in the data with `read_csv()`
        df = pd.read_csv(path, sep=',')
    
    # Display some info
    if display:
        print(f"[DATABASE]:\n{df}\n")
        print(f"[HEAD]:\n{df.head()}\n")
        print(f"[KEYS]:\n{df.keys()}\n")

    # Convert the categorical columns into numeric
    df['Class'] = le.fit_transform(df['Class'])
    #pd.options.display.max_columns = 8

    # Create Folder (overwrite if already exists one) and store new csv file
    create_folder(output, overwrite=True)
    df.to_csv(path_or_buf=output + '/dataset.csv', index=False, float_format='%.6f')

    return df


def split_data(csv_path, test_ratio=0.2, val_ratio=0.25, val=False, display=False):
    """Split data in order to have train set, validation set and test set 
    
    # Arguments
        - csv_path: the file path of the csv to read in 
        - test_ratio: value to split the original set to train and test; default is 20%.
        - val_ratio: value to split train set to train and validation set; dafault is 25%
        - val: boolean for creating validation set
        - display: boolean for print more info 
    # Return
        - 
    """

    # Load df from preprocess function
    df = preprocess(csv_path, display=display)

    # Dataset
    cols = [col for col in df.columns if col not in ['Class']]
    data = df[cols]

    # Label
    target = df['Class'] 

    # Display some info
    if display:
        print(f"\n[DATA]:\n {data}\n")
        print(f"[LABEL]:\n {target}")

    # Split dataset into train and test sets
    # ‘random_state’ just ensures that we get random results every time
    # x -> data y -> label 
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = test_ratio, random_state = 10)

    if val:
        # Split train set in order to have a validation set 
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_ratio, random_state=10)

        return x_train, y_train, x_test, y_test, x_val, y_val

    else:
        return x_train, y_train, x_test, y_test


def create_folder(save_path, overwrite=False):
    """ Check if the directory exists, delete it if any and then create a new one
    
    # Arguments
        - save_path: the path of the folder that must be created
        - overwrite: boolean for deleting or not if folder already exists
    """

    # checks if the folder already exists
    if (os.path.exists(save_path)):
        # delete the folder if already exists 
        if overwrite:
            shutil.rmtree(save_path)
        else:
            print (f"[INFO] Folder {save_path} already exists and will not be overwritten")
            return
            
    # Create the output folder
    try:
        os.mkdir(save_path)
        print (f"[INFO] Successfully created the directory {save_path}")
    except OSError as e:
        print (f"[INFO] Creation of the directory {save_path} failed: {e}")


def get_datetime():
    """Return a string containing datetime info 
    """
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("[%b-%d-%Y][%H:%M:%S]")
    return dt_string