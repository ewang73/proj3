#Sheena Ganju, CS 4641 HW 1
#Decision Trees

#info from http://scikit-learn.org/stable/modules/
#generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans

#import sklearn statements
import sklearn as sklearn
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

#for graph from http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

#k-means specific imports 
from sklearn import mixture
from sklearn.metrics import accuracy_score

#other imports 
import scikitplot as skplt
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import time
import itertools
from scipy import linalg
from sklearn import datasets
from sklearn.model_selection import validation_curve
from datetime import date

#Read data in using pandas
# trainDataSet = pd.read_csv("london_crime_by_lsoa.csv", sep = ',', header = None, low_memory = False)
# print("data read in")
# #encode text data to integers using getDummies
# traindata = pd.get_dummies(trainDataSet)
# traindata = traindata[:1000]
# Create decision Tree using major_category, month, year, to predict violent or not 
# train split uses default gini node, split using train_test_split

# X = traindata.values[1:, 1:]
# Y = traindata.values[1:,0]

# iris = datasets.load_iris()
# X = iris.data
# Y = iris.target

# traindata = X

traindata = pd.read_csv("bcICA.csv", sep = ',', header = None, low_memory = False)
Y = traindata.values[1:,8]
traindata = traindata.values[1:, 0:8]
X = traindata

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title(title)

compnum = [2, 3, 4, 6, 8, 10]
for each in compnum: 
    t0= time.clock()

    # Fit a Gaussian mixture with EM using n components
    gmm = mixture.GaussianMixture(n_components= each, covariance_type='full')
    
    gmm = gmm.fit(traindata)
    # print(gmm.means_)
    print(gmm.converged_)
    print("Lower Bound: ")
    print(gmm.lower_bound_)
    t1= time.clock()
    timetaken = str(t1-t0)
    print("Computation Time: " + timetaken)
    plot_results(traindata, gmm.predict(traindata), gmm.means_, gmm.covariances_, 0,
                'Gaussian Mixture')

    dpgmm = mixture.BayesianGaussianMixture(n_components=each,
                                        covariance_type='full').fit(traindata)
    plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
                    'Bayesian Gaussian Mixture with a Dirichlet process prior')
    plt.show()

# learning curve purposes
gmm = mixture.GaussianMixture(n_components= 10, covariance_type='full')
cv = train_test_split(X, Y, test_size=.30, random_state= 20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30, random_state= 20)

##Y_prediction = gmm.predict(X_test)
##
##from sklearn.metrics import log_loss
##loss = log_loss(Y_test, Y_prediction)*100
##print("Loss: " + str(loss))

skplt.estimators.plot_learning_curve(gmm, X, Y, title = "Learning Curve: GMM")
plt.show()