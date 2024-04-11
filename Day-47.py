# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:22:35 2023

@author: Dnyaneshwari....
"""

#PCA algorithm step by step
#allpication:to increase response time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

#Step-1
#getting data selected
marks=np.array([[3,4],[2,8],[6,9]])
marks
df=pd.DataFrame(marks,columns=['X','Y'])
df
#
plt.scatter(x=df['X'],y=df['Y'])

#Step-2  Scaling of data marix , normalization of data matrix
mean_by_column=np.mean(df.T,axis=1)
mean_by_column
#X    3.666667
#Y    7.000000

scaled_data=df-mean_by_column
scaled_data
'''
X    Y
0 -0.666667 -3.0
1 -1.666667  1.0
2  2.333333  2.0
'''
#calulated Transpose matrix of given matrix
df.T
'''
X    Y
0 -0.666667 -3.0
1 -1.666667  1.0
2  2.333333  2.0
'''
#Step-3 find covarience matrix of scaled data
cov_mat=np.cov(scaled_data.T)
cov_mat
'''
array([[4.33333333, 2.5       ],
       [2.5       , 7.        ]])
'''

#Step-4
Eval,Evec=eig(cov_mat)
Eval #Eigen Values are:-   array([2.83333333 , 8.5])
Evec #Eigen vector is below
'''
array([[-0.85749293, -0.51449576],
       [ 0.51449576, -0.85749293]])
'''

#Step-5
#Get orginal data projected on principle components
#Axis shifting phase
projected_Data=Evec.T.dot(scaled_data.T)
projected_Data.T
'''
array([[-9.71825316e-01,  2.91547595e+00],
       [ 1.94365063e+00,  1.11022302e-16],
       [-9.71825316e-01, -2.91547595e+00]])
'''
########################################################


#PCA algorithm anlysis with sklearn

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit_transform(df)
'''
array([[ 2.91547595e+00, -9.71825316e-01],
       [-6.86635020e-16,  1.94365063e+00],
       [-2.91547595e+00, -9.71825316e-01]])
'''

################################################################

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
uni=pd.read_excel("C:/Datasets/University_Clustering1.xlsx")
uni
uni=pd.read_csv("C:/Datasets/University_Clustering.csv")
uni.dtypes

uni.describe()
uni.columns
uni=uni.drop({'Expenses'},axis=1)

#considering only numeric data
uni.data=uni.iloc[:,2:]
#Normalising the Numerical Data
uni_normal=scale(uni.data)
uni_normal
#change datattype of Expenses
#ValueError: could not convert string to float: '22, 704'

pca=PCA(n_components=2)
pca_values=pca.fit_transform(uni_normal)

#
var = pca.explained_variance_ratio_
var

#commulative var
var1 = np.cumsum(np.round(var,decimals=4)*100)
var1
#var plot for pca components 
plt.plot(var1,color='red')
#pca scores
pca_values

pca_data=pd.DataFrame(pca_values)
pca_data.columns='comp0','comp1'
#this is Univ column of uni dataframe
final_dia=pd.concat([uni.Univ,pca_data.iloc[:,0:3]],axis=1)


#scatter diagram
#TypeError: ufunc 'isfinite' not supported for the input types, 
#and the inputs could not be safely coerced to any supported 
#types according to the casting rule ''safe''

ax=final_dia.plot(x='comp0',y='comp1',kind='scatter',figsize=(12,8))
final_dia[['comp0','comp1','Univ']].apply(lambda x: ax.text(*x),axis=1)

##################################################################