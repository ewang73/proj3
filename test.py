#import sklearn statements
import sklearn as sklearn
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

#for graph from http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

#other imports 
import scikitplot as skplt
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import time
from sklearn import datasets
from sklearn.model_selection import validation_curve
from datetime import date
from sklearn.preprocessing import StandardScaler


arr = [0.44272026, 0.18971182, 0.09393163, 0.06602135, 0.05495768, 0.04024522, 0.02250734, 0.01588724]
sum = 0
for i, each in enumerate(arr):
    sum += each
    print(f"{i+1}: {str(sum)}")


# traindata = datasets.load_iris()['data']

# z_scaler = StandardScaler()
# z_data = z_scaler.fit_transform(traindata)

# result = PCA().fit(z_data)

# print("Components"+ str(result.components_))
# print("Explained Variance"+ str(result.explained_variance_))
# print("Explained Variance Ration" + str(result.explained_variance_ratio_))
# print("PCA Score"+ str(result.score(traindata, y= None)))
# plt.semilogy(result.explained_variance_ratio_, '--o')
# plt.semilogy(result.explained_variance_ratio_.cumsum(), '--o')
# plt.show()


# IRIS PCA
"""
=== Run information ===

Evaluator:    weka.attributeSelection.PrincipalComponents -R 0.95 -A 5
Search:       weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1
Relation:     irisORIG
Instances:    150
Attributes:   5
              150
              4
              setosa
              versicolor
              virginica
Evaluation mode:    evaluate on all training data



=== Attribute Selection on all input data ===

Search Method:
	Attribute ranking.

Attribute Evaluator (unsupervised):
	Principal Components Attribute Transformer

Correlation matrix
  1     -0.12   0.87   0.82 
 -0.12   1     -0.43  -0.37 
  0.87  -0.43   1      0.96 
  0.82  -0.37   0.96   1    


eigenvalue	proportion	cumulative
  2.9185 	  0.72962	  0.72962	-0.58setosa-0.565versicolor-0.521150+0.2694
  0.91403	  0.22851	  0.95813	0.9234+0.377150+0.067versicolor+0.024setosa

Eigenvectors
 V1	 V2	
-0.5211	 0.3774	150
 0.2693	 0.9233	4
-0.5804	 0.0245	setosa
-0.5649	 0.0669	versicolor

Ranked attributes:
 0.2704  1 -0.58setosa-0.565versicolor-0.521150+0.2694
 0.0419  2 0.9234+0.377150+0.067versicolor+0.024setosa

Selected attributes: 1,2 : 2
"""

# IRIS infogain
"""
=== Run information ===

Evaluator:    weka.attributeSelection.InfoGainAttributeEval 
Search:       weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N 2
Relation:     irisORIG
Instances:    150
Attributes:   5
              150
              4
              setosa
              versicolor
              virginica
Evaluation mode:    evaluate on all training data



=== Attribute Selection on all input data ===

Search Method:
	Attribute ranking.

Attribute Evaluator (supervised, Class (nominal): 5 virginica):
	Information Gain Ranking Filter

Ranked attributes:
1.418  3 setosa
1.378  4 versicolor

Selected attributes: 3,4 : 2
"""