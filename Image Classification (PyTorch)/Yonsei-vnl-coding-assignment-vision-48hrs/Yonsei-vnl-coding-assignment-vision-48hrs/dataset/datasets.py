from numpy import *
# import util

class C100Dataset:
    """
    X is a feature vector
    Y is the predictor variable
    """
    tr_x = None  # X (data) of training set.
    tr_y = None  # Y (label) of training set.
    ts_x = None # X (data) of test set.
    ts_y = None # Y (label) of test set.

    def __init__(self, filename):
        ## read the csv for dataset (cifar100.csv, cifar100_lt.csv or cifar100_nl.csv), 
        # 
        # Format:
        #   image file path,classname
        
        ### TODO: Read the csv file and make the training and testing set
        ## YOUR CODE HERE

        ### TODO: assign each dataset
        tr_x = None  ### TODO: YOUR CODE HERE
        tr_y = None  ### TODO: YOUR CODE HERE
        ts_x = None ### TODO: YOUR CODE HERE
        ts_y = None ### TODO: YOUR CODE HERE

    def getDataset(self):
        return [self.tr_x, self.tr_y, self.ts_x, self.ts_y]