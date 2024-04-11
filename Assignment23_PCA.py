# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 20:13:36 2023

@author: Dnyaneshwari...
"""

#Problem Statement:
'''
#

Perform hierarchical and K-means clustering on the dataset. 
After that, perform PCA on the dataset and extract the first 3 
principal components and make a new dataset with these 3 principal 
components as the columns. Now, on this new dataset, perform 
hierarchical and K-means clustering. Compare the results of 
clustering on the origiHANGINGl dataset and clustering on the principal 
components dataset (use the scree plot technique to obtain the 
optimum number of clusters in K-means clustering and check if 
you’re getting similar results with and without PCA).


'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df=pd.read_csv("C:/Datasets/Transaction_Retail.csv")
df
df.describe()
df.columns
df.dtypes
#################################################################

#1.Business objectives
#To perform clustering  and KMeans for EastWestAirlines1 .
#To draw the inference.

#2.Business Constraints


##################################################################

#2. Work on each feature of the dataset to create a data dictioHANGINGry 
#as displayed in the below image:

df.columns
dic={   'Feature_HANGINGme':['HEART', 'T-LIGHT', 'WHITE', 
       'HOLDER', 'HANGING'],
         'Description':'Columns',
         'Type':['Quantitative','NomiHANGINGl'],
         'Relevence':'Irrelevent'
     }
new_df=pd.DataFrame(dic)
#all array must be of same length
new_df
###################################################################

#3. Data Pre-processing 
#Data Cleaning, Feature Engineering, etc.

df.head()
#2 FINDING DUPLICATES
#drop 
duplicates=df.duplicated()
duplicates
#output is a single column it present true otherwise false.
sum(duplicates)#552814
#so 552814 duplicates are present

#3. OUTLIERS AHANGINGLYSIS
sns.boxplot(df.HANGING)
sns.boxplot(df.NA)
IQR=df.HANGING.quantile(0.75)-df.HANGING.quantile(0.25)
IQR
#73876.5

lower_limit=df.HANGING.quantile(0.75) - 1.5*IQR
lower_limit
#-18410.75
upper_limit=df.HANGING.quantile(0.75) + 1.5*IQR
upper_limit
#203218.75

######################################################


#OUTLIER TREATMENT
#TRIMMING

outliers_df=np.where(df.HANGING > upper_limit,True, np.where(df.HANGING<lower_limit,True,False)) 
outliers_df
df_trimmed=df.loc[~ outliers_df]
df_trimmed
df.shape
#(3999, 12)
df_trimmed.shape
#(3733, 12)
#therefore there are 266 outliers that is trimmed

#REPLACEMENT TECHQUIES

df_replaced=pd.DataFrame(np.where(df.HANGING > upper_limit , upper_limit,np.where(df.HANGING < lower_limit , lower_limit,df.HANGING)))
#if values are greter than upper limit mapped to the upper limit
#if values are lower than lower limit mapped to the lower limit

sns.boxplot(df_replaced[0])

#Winsorizer
from feature_engine.outliers import Winsorizer

winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['HANGING'])

df_t=winsor.fit_transform(df[{'HANGING'}])
sns.boxplot(df['HANGING'])
sns.boxplot(df_t['HANGING'])


###################################################################


#4. Exploratory Data AHANGINGlysis (EDA):
#4.1. Summary.
#4.2. Univariate aHANGINGlysis.
#4.3. Bivariate aHANGINGlysis.


df.columns
df.shape
#(3999, 12)

df["HANGING"].value_counts()
df["ItalCook"].value_counts()
df["Qual_miles"].value_counts()
df["cc1_miles"].value_counts()
df["cc2_miles"].value_counts()
df["cc3_miles"].value_counts()
df["HANGING"].value_counts()
df["Flight_miles_12mo"].value_counts()
df["Flight_trans_12"].value_counts()
df["Days_since_enroll"].value_counts()
df["Award?"].value_counts()

# the given dataset is a imHANGINGd dataset

###################################################################
#scatter plot
df.plot(kind='scatter', x='HANGING', y='HANGING') ;
plt.show()
#2D scatter plot
sns.set_style("whitegrid");
sns.FacetGrid(df, hue="WHITE").map(plt.scatter, "HANGING", "HANGING").add_legend();
plt.show();
#pair plot
sns.pairplot(df, hue="HANGING");

#########################################################

#Mean, Variance, Std-deviation,  
print("Means:")
print(np.mean(df["HANGING"]))
#Mean with an outlier.
print(np.mean(np.append(df["HANGING"],50)));
print(np.mean(df["HANGING"]))
print(np.mean(df["HANGING"]))

print("\nStd-dev:")
print(np.std(df["HANGING"]))
print(np.std(df["HANGING"]))
print(np.std(df["HANGING"]))

print("\nMedians:")
print(np.median(df["HANGING"]))

#####################################################################
'''
5. Model Building 
5.1 Build the model on the scaled data (try multiple options).
5.2 Perform the hierarchical clustering and visualize the 
clusters using dendrogram.
5.3 Validate the clusters (try with different number of 
clusters) – label the clusters and derive insights 
(compare the results from multiple approaches).

'''

from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering 


z=linkage(df, method='complete',metric='euclidean') 
plt.figure(figsize=(15,8))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')

#ref help of dendrogram 
#sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#dendrogram()
#applying agglomerative clustering choosing 3 as clusters 
#from dendrogram 
#whatever has been displayed in dendrogram is not clustering 
#It is just showing number of possible clusters 
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df)

#apply labels to the clusters 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
df['HANGING'] = cluster_labels 
#we want to relocate the column 7 to 0th position 
df = df.iloc[:,[7,1,2,3,4,5,6]]
#now check the Univ1 datafraame 
df.iloc[:,2:].groupby(df.HANGING).mean()



##########################################################
'''
6.MODEL BULDING
6.1 Build the model on the scaled data (try multiple options).
6.2 Perform K- means clustering and obtain optimum number of 
clusters using scree plot.
6.3 Validate the clusters (try with different number of clusters) 
– label the clusters and derive insights (compare the results 
from multiple approaches).


'''
#KMeans 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

new_df=pd.read_csv("C:/Datasets/Transaction_Retail.csv")
new_df.dtypes
new_df.describe()
new_df.columns
new_df=new_df.drop({'HANGING'},axis=1)


def norm_fun(i):
    x=(i-i.min()) / (i.max()-i.min())
    return x

df_norm=norm_fun(new_df.iloc[:,1:])
#what will be the cluster number,will it be 1,2,3,4....

TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=1)
    kmeans.fit(df_norm)
    
    TWSS.append(kmeans.inertia_)
    


TWSS
''' TWSS Values
[1816.8234864503236,
 1816.8234864503236,
 1816.8234864503236,
 1816.8234864503236,
 1816.8234864503236,
 1816.8234864503236]
'''
plt.plot(k,TWSS,'bo-');
plt.xlabel("No of Clusters(K)");
plt.ylabel("Total_within_SS(TWSS)")

############################################################

model=KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
mb=pd.Series(model.labels_)
new_df['HANGING']=mb
new_df.head()
new_df=new_df.iloc[:,[7,0,1,2,3,4,5,6]]
new_df
new_df.iloc[:,2:8].groupby(new_df.HANGING).mean()
new_df.to_csv("C:/Datasets/Transaction_Retail.csv",encoding='utf-8')

################################################################

#7. Write about the benefits/impact of the solution - 
#in what way does the business (client) benefit from the solution provided?


#########################################################################################

#PCA algorithm step by step
#allpication:to increase response time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.liHANGINGlg import eig

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
#Get orgiHANGINGl data projected on principle components
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
uni=pd.read_excel("C:/Datasets/Transaction_Retail.csv")
uni
uni=pd.read_csv("C:/Datasets/Transaction_Retail.csv")
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
fiHANGINGl_dia=pd.concat([uni.Univ,pca_data.iloc[:,0:3]],axis=1)


#scatter diagram
#TypeError: ufunc 'isfinite' not supported for the input types, 
#and the inputs could not be safely coerced to any supported 
#types according to the casting rule ''safe''

ax=fiHANGINGl_dia.plot(x='comp0',y='comp1',kind='scatter',figsize=(12,8))
fiHANGINGl_dia[['comp0','comp1','Univ']].apply(lambda x: ax.text(*x),axis=1)

##################################################################
