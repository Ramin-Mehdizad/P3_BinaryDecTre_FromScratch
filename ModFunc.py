

"""
===============================================================================
 Created on Feb 20, 2020

 @author: Ramin Mehdizad Tekiyeh

 This code is written in Python 3.7.4 , Spyder 3.3.6
===============================================================================
"""


#==============================================================================
# importing standard classes
#==============================================================================
#import logging
import numpy as np
import pandas as pd
import os
import csv
import argparse

 
#==============================================================================
# importing module codes
#==============================================================================
import ModVar as Var


#==============================================================================
# importing graphviz
#==============================================================================
if Var.graphviz_import_ID==1:
    try:
        import graphviz
    except:
        print("!!! Graphviz couldn't be imported !!!")
        Var.graphviz_import_ID=0


#==============================================================================
# this function asks user input data
#==============================================================================
def Call_Parser():
    
    # create parse class
    parser1 = argparse.ArgumentParser(add_help=True,prog='Binary ID3 Decision Tree from Scratch',
             description='* This program runs binary ID3 decision tree *')
    
    # set program version
    parser1.add_argument('-v','--version',action='version', version='%(prog)s 1.0')
   
    # script path
    Var.MainDir=os.path.abspath(os.getcwd())
    print('MainDir', Var.MainDir)
    
    
    # which dataset to use
    parser1.add_argument('-d', '--DatasetNo', action='store', 
                         default='1',  dest='DatasetNo', choices=['1', '2', '3'],
                         help='1: Dataset1(Simple)     2: Dataset2(Moderate)   3: Dataset3(Complex)')
    
    # set train Data file address
    parser1.add_argument('-m', '--MaxDepthLowLimit', action='store',
                        default=1, dest='MaxDepthLowLimit', 
                        help='Enter lower limit of Maxdepth')
    
    # set train Data file address
    parser1.add_argument('-n', '--MaxDepthHighLimit', action='store',
                        default=10, dest='MaxDepthHighLimit', 
                        help='Enter higher limit of Maxdepth')
    
    # whether to save plots or not
    parser1.add_argument('-p', '--SavePlot', action='store', 
                         default='1',  dest='SavePlot', choices=['0', '1'],
                         help='0: Dont Save plots     1: Save plots')

    #  print gain in each split 
    parser1.add_argument('-g', '--PrintCalcDetails', action='store',
                         default='1', dest='PrintCalcDetails', choices=['0', '1'],
                         help='0: Don"t print Gain     1: print Gain')
    
    #  print gain in each split 
    parser1.add_argument('-s', '--PrintSplitData', action='store',
                         default='1', dest='PrintSplitData', choices=['0', '1'],
                         help='0: Don"t print split data     1: print split data')
    
    # whether to create log file or not
    parser1.add_argument('-l', '--log', action='store',
                         default='1', dest='logFile', choices=['0', '1'],
                         help='0: Dont write logfile     1: write logfile')

    # indicates when to exit while loop
    entry=False
    while entry==False:
        
        # --------------in this section we try to parse successfully-----------
        # initialize
        ParsErr=0
        PathErr=0
        TrainFileErr=0
        TestFileErr=0
        
        # function to call input data from command line    
        Show_Input_Message()
        
        # user wanted to continue with default values
        if Var.str_input=='':
            # it means that we want to continue with default parameters
            Var.args=parser1.parse_args()
            # exit while loop
            entry=True
        elif Var.str_input.upper()=='Q':
            # exit script
            sys.exit()
        else:
            entry=True
            ParsErr=0
            try:
                Var.args=parser1.parse_args(Var.str_input.split(' '))
            except:
                entry=False
                ParsErr=1
        #----------------------------------------------------------------------
        

        #-------------After having parsed successfully, we coninue-------------
        # continue if parse was done successfully
        if ParsErr==0:  
            # check if train data base file exists
            PathErr=0
            if not(os.path.exists(Var.MainDir+'\datasets')):
                PathErr=1
                entry=False
                print("datasets path doesn't exist.")
                print('Enter a valid path.')
                
            # continue if train\test data file exists
            if PathErr==0:  
                Var.trn_name= '{}\datasets\dataset_{}.train'.format(
                                            Var.MainDir, Var.args.DatasetNo)
                Var.tst_name='{}\datasets\dataset_{}.test'.format(
                                            Var.MainDir, Var.args.DatasetNo)
                
                print('\ntrn_name is: \n', Var.trn_name)
                print('\ntst_name is: \n', Var.tst_name)
                
                #check if  data file exists
                if not(os.path.isfile(Var.trn_name)):
                    TrainFileErr=1
                    entry=False
                    print("Train Data file address doesn't exist.")
                    print("Enter a valid file address. ")
                     
                #check if test data base file exists
                if not(os.path.isfile(Var.tst_name)):
                    TestFileErr=1
                    entry=False
                    print("Test Data file address doesn't exist.")
                    print("Enter a valid file address. ")
  
                # here everything about input data is correct so we 
                # continue to execution
                if  TrainFileErr==0 and TestFileErr==0:
                    # read train and test data
                    Read_Data()
                    entry=True
                    
                    # settitng flags
                    Var.FlagSavePlot=True if Var.args.SavePlot=='1' else False
                    Var.FlagPrintCalcDetails=True if Var.args.PrintCalcDetails=='1' else False
                    Var.FlagPrintSplitData=True if Var.args.PrintSplitData=='1' else False
                    
                    # print successfully parsed data
                    PrintParsedData()
            else:
                continue
                    
        #----------------------------------------------------------------------


#==============================================================================
# this function asks user input data
#==============================================================================  
def Show_Input_Message():
    
    global str_input
    print('')
    print('|===========================================================')
    print('|  ==> To run the code with default values, just press Enter')
    print('|  ==> Otherwise:')
    print('|  ==> Enter the parameters as following format:')
    print('|')
    print('|  -d 1 -m 3 -n 10 -p 0 -g 1 -s 1-l 1')
    print('|')
    print('|  ==> To get help, type "-h" and press Enter')
    print('|  ==> To exit program, type "Q" and press Enter')
    print('|===========================================================')
    
    Var.str_input=input('  Enter parameters: ').strip()
    

#==============================================================================
# this function gets computer spec
#==============================================================================
def GetSysInfo():
    import platform,socket,re,uuid,psutil
    try:
        Var.sysinfo.append(['platform',platform.system()]) 
        Var.sysinfo.append(['platform-release',platform.release()])
        Var.sysinfo.append(['platform-version',platform.version()])
        Var.sysinfo.append(['architecture',platform.machine()])
        Var.sysinfo.append(['hostname',socket.gethostname()])
        Var.sysinfo.append(['ip-address',socket.gethostbyname(socket.gethostname())])
        Var.sysinfo.append(['mac-address',':'.join(re.findall('..', '%012x' % uuid.getnode()))])
        Var.sysinfo.append(['processor',platform.processor()])
        Var.sysinfo.append(['ram',str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"])
    
    except Exception as e:
        print(e)


#==============================================================================
# this function reads input data
#============================================================================== 
def Read_Data():

    # Load  train data
    f1 = np.genfromtxt(Var.trn_name, missing_values=0, skip_header=0, 
                                               delimiter=',', dtype=int)
    Var.y_train = f1[:, 0]
    Var.X_train = f1[:, 1:]
    
    # Load test data
    f2 = np.genfromtxt(Var.tst_name, missing_values=0, skip_header=0, 
                                               delimiter=',', dtype=int)
    Var.y_test = f2[:, 0]
    Var.X_test = f2[:, 1:]
    
    #--------------------------------------------------------------------------
    # test with sample data
    # x0=np.array([0,0,1,2,2,2,1,0,0,2,0,1,1,2]).reshape(-1,1)
    # x1=np.array([0,0,0,1,2,2,2,1,2,1,1,1,0,1]).reshape(-1,1)
    # x2=np.array([3,3,3,3,2,2,2,3,2,2,2,3,2,3]).reshape(-1,1)
    # x3=np.array([0,1,0,0,0,1,1,0,0,0,1,1,0,1]).reshape(-1,1)
    # Var.X_train=np.hstack((x0,x1,x2,x3))
    # Var.y_train=np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0]).reshape(-1,1)
    # print(Var.X_train)
    # print(Var.y_train)
    
    # test with sample data
    # x0=np.array([0,0,1,2,2,2,1,0,0,2,0,1,1,2]).reshape(-1,1)
    # x1=np.array([0,0,0,1,2,2,2,1,2,1,1,1,0,1]).reshape(-1,1)
    # Var.X_trn=np.hstack((x0,x1))
    # Var.y_trn=np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0]).reshape(-1,1)
    # print(Var.X_train)
    # print(Var.y_train)
    #--------------------------------------------------------------------------

    
#==============================================================================
# this function calculates attribute value pair list
#==============================================================================
def attr_val_pair_list_calc(X_trn):
    attr_val_pair_list=list()
    # Num_Feat=np.shape(X_trn)[1]
    for i in range(np.shape(X_trn)[1]):
        AttrVal_list=list()
        for _ , val in enumerate(X_trn[:,i]):
            if val not in AttrVal_list: AttrVal_list.append(val)
        # sort it so that it goes in order
        AttrVal_list=np.sort(AttrVal_list)
        for j in AttrVal_list:
            attr_val_pair_list.append((i,j))
    return(attr_val_pair_list)


#==============================================================================
# this function plots tree in console
#==============================================================================
def Treeplot(tre,dep,indic,indent):
    if dep==0: 
        print('Tree')
        print('|')
    dep+=1
    
    for i , DictItem in enumerate(tre):
        TruFls=str(DictItem[2])[0]
        ttl='[{},{},{}]'.format(str(DictItem[0]),
                                    str(DictItem[1]),TruFls)
        print((dep-1)*indent+indic+str(ttl))
        
        if type (tre[DictItem])==int:
            # Base Case
            print((dep)*indent+str(tre[DictItem]))
        else:
            # Recursive Case
            Treeplot(tre[DictItem],dep,indic,indent)


#==============================================================================
# this function Converts a tree to DOT format for use with visualize/GraphViz
#==============================================================================
def to_graphviz(tree, dot_string='', uid=-1, depth=0):

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="Attr{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


#==============================================================================
# Uses GraphViz to render a dot file
#==============================================================================
def render_dot_file(dot_string, save_file, image_format='png'):

    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    try:
        os.environ["PATH"] += os.pathsep + 'C:/Users/Home/Anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin'
        graph = graphviz.Source(dot_string)
        graph.format = image_format
        if Var.FlagSavePlot: graph.render(save_file, view=True) 
        if Var.logFlag: My_Log.TreePlottedByGraphviz('M')
    except:
        print('Error reaching graphviz path')   


#==============================================================================
# this function computes error
#==============================================================================
def compute_error(y_true, y_pred):
    # define positive and negative label
    Pos,Neg=1,0

    TP,FP,FN,TN=0,0,0,0

    for i in range(len(y_true)):
        if y_true[i]==Pos and y_pred[i]==Pos: TP+=1
        if y_true[i]==Neg and y_pred[i]==Pos: FP+=1
        if y_true[i]==Pos and y_pred[i]==Neg: FN+=1
        if y_true[i]==Neg and y_pred[i]==Neg: TN+=1
    conf_mat=[[TP,FN],[FP,TN]]
    Acc=(TP+TN)/(TP+TN+FP+FN)
    print('\n conf_mat is:  \n ', conf_mat)
    return Acc


#==============================================================================
# this function prinrtsparsed data
#==============================================================================
def PrintParsedData(): 
    print('') 
    print('  ========================Parsed  Data=====================')  
    print('  ', Var.args)
    print('')
    print('  Dataset No to be analyzed  =', Var.args.DatasetNo)
    print('  Min Tree Depth             =', Var.args.MaxDepthLowLimit)
    print('  Max Tree Depth             =', Var.args.MaxDepthHighLimit)
    print('  Save Plots                 =', Var.args.SavePlot)
    print('  Print calculations         =', Var.args.PrintCalcDetails)
    print('  Print split data           =', Var.args.PrintSplitData)
    print('  Create Log File            =', Var.args.logFile)
    print('  =========================================================')
    print('')
    
    











