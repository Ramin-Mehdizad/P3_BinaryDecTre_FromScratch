
"""
===============================================================================
 Created on Feb 20, 2020

 @author: Ramin Mehdizad Tekiyeh

 This code is written in Python 3.7.4 , Spyder 3.3.6
===============================================================================
"""


#==============================================================================
# deleting variables before starting main code
#==============================================================================
try:
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
except:
    print('Couldn"t erase variables from catche')
        

#==============================================================================
# imports
#==============================================================================
from sklearn.svm import SVC
import time
import argparse
import sys
import numpy as np
import os
import matplotlib.pyplot as plt


#==============================================================================
# importing module codes
#==============================================================================
import ModClass as Clss
import ModFunc as Func
import ModVar as Var

try:
    import graphviz
except:
    print("!!! Graphviz couldn't be imported !!!")
    Var.graphviz_import_ID=0
 

#==============================================================================
# main program
#==============================================================================
if __name__ == '__main__':
    
    # call input data from user by means of parsing
    Func.Call_Parser()    
    
    # create log object
    if Var.logFlag: 
        My_Log=Clss.LogClass()
        # log the data of previous lines
        My_Log.ProgStart('LM')
        My_Log.ParsedData('M')
        
    # logging system information
    Func.GetSysInfo()
    if Var.logFlag: My_Log.SysSpec(Var.sysinfo,'M')
    
    # create a range of maxdepth to be used in the loop as the tree's maxdepth
    depth_list=np.arange(int(Var.args.MaxDepthLowLimit),int(Var.args.MaxDepthHighLimit)+1)
    print('depth_list  ',depth_list)

    # each iteration of the following loop is solves the project with a 
    # tree of specific max depth 
    for Var.MaxDepth in depth_list:
    
        #new tree started
        if Var.logFlag: My_Log.NewTreStarted(Var.MaxDepth,'M')    
    
        # extract all possible pair list for binary tree
        Var.attr_val_pair_list=Func.attr_val_pair_list_calc(Var.X_train)
        
        # initialize depth
        curr_depth=0
        
        # create a classifier object
        Dec_Tree_Model_1 = Clss.Bin_ID3(Var.X_train, Var.y_train, 
                                        curr_depth, max_dep=Var.MaxDepth)
        if Var.logFlag: My_Log.NewTreClassifierCreated('M')
        
        # train the model
        Dec_Tree_Model_1.fit()
        print('tree type is:',type(Dec_Tree_Model_1.TrainedTree))
        if Var.logFlag: My_Log.NewTreTrained('M')
    
        # Pretty print it to console
        dep=0
        Func.Treeplot(Dec_Tree_Model_1.TrainedTree,dep,Var.indic,Var.indent)
        if Var.logFlag: My_Log.TrePlottedToConsole('M')
    
        # Visualize the tree and save it as a PNG image for last n
        if Var.graphviz_import_ID:
            # define last maxdepth
            if Var.MaxDepth==depth_list[-1]:
                dot_str = Func.to_graphviz(Dec_Tree_Model_1.TrainedTree)
                Func.render_dot_file(dot_str, './Learned_Tree')
        else:
            print('\n Couldnt use graphviz \n')
    
        # Compute train error
        y_pred_train = [Dec_Tree_Model_1.predict(x) for x in Var.X_train]
        Acc_train = Func.compute_error(Var.y_train, y_pred_train)
        print('Train Error = {0:4.2f}%.'.format(Acc_train * 100))
        Var.Acc_train_list.append(Acc_train)
        if Var.logFlag: My_Log.TrainErrorCalculated('M')
        
        # Compute test error
        y_pred_test = [Dec_Tree_Model_1.predict(x) for x in Var.X_test]
        Acc_test = Func.compute_error(Var.y_test, y_pred_test)
        print('Test Error = {0:4.2f}%.'.format(Acc_test * 100))
        Var.Acc_test_list.append(Acc_test)
        if Var.logFlag: My_Log.TestErrorCalculated('M')
        
    plt.figure()
    plt.plot(depth_list, Var.Acc_train_list,'-*', label='train error') 
    plt.plot(depth_list, Var.Acc_test_list,'-*', label='test error')     
    plt.xlabel('Max depth')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.title('Train and Test Accuracy Error')
    
    if Var.logFlag: My_Log.AccuracyPlotCreated('M')
    
    if Var.logFlag: My_Log.ProgTerminated('M')
    














