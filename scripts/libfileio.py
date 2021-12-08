#!/usr/bin/env python3

import numpy as np
import pickle


class WrapperFileIO(object):
    """Wrapper around common Python3 I/O"""

    def __init__(self,file_path="../data/",file_name="test.pickle"):
        self.file_path = file_path
        self.file_name = file_name

    def savePickle(self,data_dict,save_file_name=None,debug=False):
        """
        Saves the dictionary data_dict to the pickle file specified by the
        save_file_name in the directory self.file_path. If save_file_name is
        none, it will load the file specificied by the class variable
        self.file_name
        """
        file_to_save = self.file_path + (self.file_name if save_file_name is None else save_file_name)
        if debug: print("Saving Pickle File to:",file_to_save)
        with open(file_to_save, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return None

    def loadPickle(self,load_file_name=None,debug=False):
        """
        Loads the pickle file specified by the load_file_name in the directory
        self.file_path. If load_file_name is none, it will load the file specificied
        by the class variable self.file_name
        """
        file_to_open = self.file_path + (self.file_name if load_file_name is None else load_file_name)
        if debug: print("Loading Pickle File from:",file_to_open)
        with open(file_to_open, 'rb') as handle:
            return pickle.load(handle)

    def saveCSV(self,data_arr,save_file_name):
        np.savetxt(self.file_path+save_file_name,data_arr, delimiter=",")

if __name__ == '__main__':

    # Create an instance of the file IO wrapper
    myIO = WrapperFileIO()

    # Create an example dictionary
    a = {'hello': 'world'}

    # Save the example dictionary
    myIO.savePickle(a,debug=True)

    # Laod the dictionary
    b = myIO.loadPickle(debug = True)

    # Verify Equivalency
    if(a == b):
        print("Dictionaries are equivalent")
    else:
        print("Pickle broke the dictionary!")
    print(a)
    print(b)
