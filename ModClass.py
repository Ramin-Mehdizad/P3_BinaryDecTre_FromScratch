

"""
===============================================================================
 Created on Feb 20, 2020

 @author: Ramin Mehdizad Tekiyeh

 This code is written in Python 3.7.4 , Spyder 3.3.6
===============================================================================
"""


#==============================================================================
# This module contains all the Classes that are used in the main code
#==============================================================================


#==============================================================================
# importing standard classes
#==============================================================================
import logging
import numpy as np
import os


#==============================================================================
# importing module codes
#==============================================================================
import ModVar as Var
import ModFunc as Func


#==============================================================================
# this class creates Binary ID3 Class
#==============================================================================
class Bin_ID3:
    
    # initializing class instance parameters
    def __init__(self,X_train, y_train, dep, max_dep=10):
        self.X_trn=X_train
        self.y_trn=y_train
        self.depth=dep
        self.max_depth=max_dep
    
    # this function extracts y labels items
    def My_Extract_y_Labels(self,y):
        Labels_list=list()
        for _,j in enumerate(y):
            if j not in Labels_list: Labels_list.append(j)
        return Labels_list

    # this function partitions x data
    def partition(self,x):
        V=list()
        dict_partitioned=dict()
        for _ ,xx in enumerate(x):
            if xx not in V: V.append(int(xx))
        
        for i in range(len(V)):
            indices_list=list()
            for j, xx in enumerate(x):
                if xx==V[i]: indices_list.append(j)
            dict_partitioned[V[i]]= indices_list 
        return dict_partitioned

    # this function calculates the entropy
    def entropy(self,x):
        x_partitioned=self.partition(x)
        if len(x_partitioned)==1:
            print('!!!!!!--- Ent=0 ---!!!!!!')
            return 0
        elif len(x_partitioned)==3:
            print('!!! Be careful: not binary partitioning !!!')
            return
        elif len(x_partitioned)==2:
            Ent,n_Tot=0,len(x)
            keys=list(x_partitioned.keys())
            n=[len(x_partitioned[keys[0]]),len(x_partitioned[keys[1]])]
            for i in range(2):
                Pi=n[i]/n_Tot
                Ent=Ent-Pi*np.log2(Pi)
            return Ent
       
    # this function calcuates the information gain 
    def Info_Gain(self,attr_no,attr_val):
        Ent_S=self.entropy(self.y_trn)
        if Var.FlagPrintCalcDetails: print('Ent_S in start of Info_Gain calc is:\n',Ent_S)
        
        print('\n\n========  New Info_Gain Calc =========')
        print('attr_no   is:  ',attr_no)
        print('attr_val   is:  ',attr_val)
        
        # initialize info_gain
        info_gain=Ent_S

        x_partitioned=self.partition(self.X_trn[:,attr_no])
        if Var.FlagPrintCalcDetails: print('x_partitioned is: ', x_partitioned)
        
        if len(x_partitioned)==1:
            # it means all values are the same so we check if this value
            # is equal to attr_val or not
            if self.X_trn[0,attr_no]==attr_val:
                # Info_gain = Ent(s)- 0 - 0
                return info_gain
            else:
                # it means that this pair does not apply for X_trn in this level
                return 0
        
        else:
            
            n_Tot=len(self.X_trn[:,attr_no])
            
            # may be attr_val is not a value in that attribute
            # so no info gain cab ne obtained by that
            if attr_val not in list(x_partitioned.keys()):
                return 0
            
            Val_ID=x_partitioned[attr_val]
            if Var.FlagPrintCalcDetails: print('Val_ID   is:  ',Val_ID)
            n_Val=len(Val_ID)
            if Var.FlagPrintCalcDetails: print('n_Val  is:  ',n_Val)
            
            NotVal_ID=[i for i in range(n_Tot) if i not in Val_ID]
            if Var.FlagPrintCalcDetails: print('NotVal_ID   is:  ',NotVal_ID)
            n_NotVal=n_Tot-n_Val
            if Var.FlagPrintCalcDetails: print('n_NotVal  is:  ',n_NotVal)
            
            # initialize and check if all vals are equal
            y_Sv_if_S=list()
            y_NotSv_if_S=list()
            
            # if len()==0 or len()=0
            y_Sv_if_S=[int(self.y_trn[i]) for i in Val_ID]
            if Var.FlagPrintCalcDetails: print('y_Sv_if_S  is:  ',y_Sv_if_S)
            Ent_Sv=self.entropy(y_Sv_if_S)
            if Var.FlagPrintCalcDetails: print('Ent_Sv   is:  ',Ent_Sv)
            
            y_NotSv_if_S=[int(self.y_trn[i]) for i in NotVal_ID]
            if Var.FlagPrintCalcDetails: print('y_NotSv_if_S  is:  ',y_NotSv_if_S)
            Ent_NotSv=self.entropy(y_NotSv_if_S)
            if Var.FlagPrintCalcDetails: print('Ent_NotSv   is:  ',Ent_NotSv)
            
            info_gain=info_gain-n_Val/n_Tot*Ent_Sv-n_NotVal/n_Tot*Ent_NotSv
            if Var.FlagPrintCalcDetails: print('info_gain of this pair   is:  ',info_gain)
                
            return info_gain

    # this function defines the majority of label items
    def y_MajorityLabel(self,y):
        
        self.y_Labels_list=self.My_Extract_y_Labels(y)
        print('y_Labels_list is: ', self.y_Labels_list)
        
        if len(self.y_Labels_list)==1:
            lbl=int(y[0])
        elif len(self.y_Labels_list)==2:
            # we return majority label
            L1=int(self.y_Labels_list[0])
            L2=int(self.y_Labels_list[1])
            n1,n2=0,0
            if Var.FlagPrintCalcDetails: print('len(y) is:  ', len(y))
            for i in range(len(y)):
                if y[i]==L1: n1+=1
                if y[i]==L2: n2+=1
            if Var.FlagPrintCalcDetails: print('n1 is:  ',n1) 
            if Var.FlagPrintCalcDetails: print('n2 is:  ',n2)
            if n1 > n2: 
                lbl=int(L1)
            elif  n1==n2 :
                lbl=int(L1)
            else:
                lbl=int(L2)
        
        print('y majority is:  ',str(lbl))        
        return lbl  
    
    # this function call the training function   
    def fit(self):
        self.TrainedTree=self.fitt()
        
    # this function does the training
    def fitt(self):    
        self.depth=self.depth+1
        print('\n\n<<<<----------id3 depth of: '+str(self.depth)+' staretd---------->>>>')
        
        # extract class labels
        self.y_Labels_list=self.My_Extract_y_Labels(self.y_trn)
        
        print('No of Data in this ID3 is:   ', len(self.y_trn))
        
        #--------------- Base Case: ending conditions
        if self.depth > self.max_depth:
            print('max n_iter reached')
            return self.y_MajorityLabel(self.y_trn)
        else:
            if len(Var.attr_val_pair_list) ==0:
                print('\n All pairs are used up')
                return  self.y_MajorityLabel(self.y_trn)
            else: 
                # I put the following if so that it wont continue into info-gain
                # function and avoid any possible errors
                if len(self.y_Labels_list)==1:
                    return  self.y_MajorityLabel(self.y_trn)
                else:
                    #--------------- Recursive Case: 
                    self.Trained=1
                    # clac info gain of all pairs 
                    Info_Gain_list=list()
                    for i in range(len(Var.attr_val_pair_list)):
                    # for i in [0]:
                        attr_no=Var.attr_val_pair_list[i][0]
                        attr_val=Var.attr_val_pair_list[i][1]
                        info_gain=self.Info_Gain(attr_no,attr_val)
                        Info_Gain_list.append(info_gain)
                    if Var.FlagPrintCalcDetails: print('\n Info_Gain_list is:  \n', Info_Gain_list)
                    
                    # select best info gain and best pair
                    max_info_gain=0
                    best_pair_ID=0
                    if len(Var.attr_val_pair_list)==1:
                        max_info_gain=Info_Gain_list[0]
                        best_pair_ID=0
                    else:
                        for i in range(len(Var.attr_val_pair_list)):
                            if Info_Gain_list[i] > max_info_gain:
                                max_info_gain=Info_Gain_list[i]
                                best_pair_ID=i
                                
                    if Var.FlagPrintCalcDetails: print('max_info_gain is:   ',max_info_gain)            
                    if Var.FlagPrintCalcDetails: print('best attr pair ID is:   ',best_pair_ID)
                    
                    # the following means we wont get any gain if we continue
                    # the process
                    if max_info_gain==0:
                        del(Var.attr_val_pair_list[best_pair_ID])
                        return self.y_MajorityLabel(self.y_trn)
                
                    # selected feature and value
                    Sel_Feat=Var.attr_val_pair_list[best_pair_ID][0]
                    Sel_Val=Var.attr_val_pair_list[best_pair_ID][1]  
                    
                    # True leaf
                    Tuple_True=(Sel_Feat, Sel_Val , True)
                    
                    # False leaf
                    Tuple_False=(Sel_Feat, Sel_Val , False)
                    if Var.FlagPrintCalcDetails: print('tuples are: ',Tuple_True,Tuple_False)
                    
                    # now we split data into True and False to continue recursion process
                    x_partitioned=self.partition(self.X_trn[:,Sel_Feat])
                    ID_True=list(x_partitioned[Sel_Val])
                    ID_False=list()
                    for i, k in x_partitioned.items():
                        if i != Sel_Val:
                            ID_False=ID_False+list(x_partitioned[i])
                            
                    # it is better to sort
                    ID_True=np.sort(ID_True)
                    ID_False=np.sort(ID_False)
                    
                    if Var.FlagPrintCalcDetails: print('ID_True is: ',ID_True)
                    if Var.FlagPrintCalcDetails: print('ID_False: ',ID_False)
              
                    if len(ID_True)==0 or len(ID_False)==0:
                        return self.y_MajorityLabel(self.y_trn)
                
                    # extract true dataset        
                    X_trn_True=self.X_trn[ID_True,:]
                    y_trn_True=self.y_trn[ID_True]
                    # extract false dataset
                    X_trn_False=self.X_trn[ID_False,:]
                    y_trn_False=self.y_trn[ID_False]
                    
                    if Var.FlagPrintSplitData: print('X_trn_True is: ',X_trn_True)
                    if Var.FlagPrintSplitData: print('y_trn_True: ',y_trn_True)
                    
                    if Var.FlagPrintSplitData: print('X_trn_False is: ',X_trn_False)
                    if Var.FlagPrintSplitData: print('y_trn_False: ',y_trn_False)
                   
                    # eliminate the calculated pair
                    del(Var.attr_val_pair_list[best_pair_ID])

                    # create tree dictionary
                    Tree= {Tuple_True:Bin_ID3(X_trn_True, y_trn_True, self.depth, max_dep=Var.MaxDepth).fitt(),
                        Tuple_False:Bin_ID3(X_trn_False, y_trn_False, self.depth, max_dep=Var.MaxDepth).fitt()}
                    
                    print('Tree is:  \n', Tree)
                    
                    return Tree
     
    # this functions callthe predicting function                
    def predict(self,x):
        tre=self.TrainedTree
        return self.pred(x,tre)
        
    # this function does the prediction
    def pred(cls,x,tre):
        # because we dont want to pass self and we want to extract tree as 
        # tree and also we need to use the pred function as recursive one,
        # we pass cls to use this function but not the self instance.
        key1=list(tre.keys())[0]
        attr=key1[0]
        val=key1[1]
        
        if x[attr]==val:
            tuple1=(attr,val,True)
            lbl=tre[tuple1]
        else:
            tuple1=(attr,val,False)
            lbl=tre[tuple1]
            
        if type(lbl) is dict:
            tre=lbl
            return cls.pred(x,tre)
        else:
            return lbl


#==============================================================================
# this class defines logging events and results into *.log file
#        
# Note:
#     All the methods and logging data are created in the methods of this class
#     Then the logging action is done in the main code
#==============================================================================
class LogClass():
    
    # initializing class instance parameters
    def __init__(self):
        # initialize
        self.logger=logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.formatter=logging.Formatter('%(message)s')
        self.filehandler=logging.FileHandler('Log.log')
        self.filehandler.setFormatter(self.formatter)
        self.logger.addHandler(self.filehandler)
        self.splitterLen=84
        self.splitterChar='*'
        self.EndSep=self.splitterChar*self.splitterLen
        
    # this method performs the format of logging for each log action    
    def LogFrmt(self,n):
        if n=='M':
            self.formatter=logging.Formatter(' %(message)s')
            self.filehandler.setFormatter(self.formatter)
        elif n=='LM':
            self.formatter=logging.Formatter('%(levelname)s: %(message)s')
            self.filehandler.setFormatter(self.formatter)
        elif n=='TLM':
            self.formatter=logging.Formatter('%(acstime)s: %(levelname)s: %(message)s')
            self.filehandler.setFormatter(self.formatter)
    
    # this method logs ParsedData
    def ParsedData(self,n):
        self.LogFrmt(n)
        title=' Data Entered by User '
        sp=self.splitterChar*(round((self.splitterLen-len(title))/2))
        self.logger.info(sp+title+sp)
        self.logger.info('')
        self.logger.info('  Results Path         ='+ os.getcwd())
        self.logger.info('  Save Plots           ='+ Var.args.SavePlot)
        self.logger.info('  Create Log File      ='+ Var.args.logFile)
        self.logger.info(self.EndSep)
    
    # this method logs start of the main
    def ProgStart(self,n):
        self.LogFrmt(n)
        self.logger.info('')
        title=' Main Program Started '
        sp=self.splitterChar*(round((self.splitterLen-len(title))/2))
        self.logger.info(sp+title+sp)
        self.logger.info('')
        self.logger.info('')
     
    # this method logs the system on which the analysis if performed  
    def SysSpec(self,sysinfo,n):
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('')
        title=' COMPUTER SPEC '
        sp=self.splitterChar*round((self.splitterLen-len(title))/2)
        self.logger.info(sp+title+sp)
        self.logger.info('Data analsys is done on the system with following spec:\n')  
        for i,[a1,a2] in enumerate(Var.sysinfo):
            DataStartChar=30
            len1=len(Var.sysinfo[i][0])
            Arrow='-'*(DataStartChar-len1)+'> '
            self.logger.info(Var.sysinfo[i][0]+Arrow+Var.sysinfo[i][1])
        self.logger.info(self.EndSep)

    # this method logs start of a new loop for a treewith specific max depth    
    def NewTreStarted(self,maxdepth,n):    
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('New Decision Tree calculation with max depth={0} started'.format(maxdepth))
        
    # this method logs new classifier object created    
    def NewTreClassifierCreated(self,n):    
        self.LogFrmt(n)
        self.logger.info('New Tree class created')

    # this method logs new tree trained    
    def NewTreTrained(self,n):    
        self.LogFrmt(n)
        self.logger.info('Decision Tree trained')
        
    # this method logs tree plotted to console     
    def TrePlottedToConsole(self,n):    
        self.LogFrmt(n)
        self.logger.info('Tree plotted toconsole')

    # this method logs tree plotted by graphviz     
    def TreePlottedByGraphviz(self,n):    
        self.LogFrmt(n)
        self.logger.info('Tree plotted by graphviz and the plot saved')
        
    # this method logs train error calculated     
    def TrainErrorCalculated(self,n):    
        self.LogFrmt(n)
        self.logger.info('Train error calculated') 
        
    # this method logs test error calculated     
    def TestErrorCalculated(self,n):    
        self.LogFrmt(n)
        self.logger.info('Test error calculated')
        
    # this method logs accuracy plot created     
    def AccuracyPlotCreated(self,n):    
        self.LogFrmt(n)
        self.logger.info('Test error calculated')
        
    # this method logs accuracy plot created     
    def ProgTerminated(self,n):    
        self.LogFrmt(n)
        self.logger.info('')
        self.logger.info('Program terminated successfully')
        
        
        
        
        

















