
"""
===============================================================================
 Created on Feb 20, 2020

 @author: Ramin Mehdizad Tekiyeh

 This code is written in Python 3.7.4 , Spyder 3.3.6
===============================================================================
"""


#==============================================================================
# argparse vars
#==============================================================================
str_input=''
args=''
MainDir=''
args=''
sysinfo=[]


#==============================================================================
# parameters used in pretty printing the tree
#==============================================================================
ind_len=8
indic_len=5
indent='|' + ind_len * ' '
indic='+' + indic_len * '='


#==============================================================================
# data sets
#==============================================================================
y_train=[]
X_train=[]
y_test=[]
X_test=[]


#==============================================================================
# flags
#==============================================================================
# graphviz_import_ID should be 1 because when importing miodules the whole 
# imports are done, so we should do the imports when the modules are imported
graphviz_import_ID=1   
FlagSavePlot=False
PrintCalcDetails=False
FlagPrintSplitData=False
logFlag=1


#==============================================================================
# other variables
#==============================================================================
attr_val_pair_list=[]

trn_name=[]
tst_name=[]

Acc_train_list=list()
Acc_test_list=list()

MaxDepth=0







