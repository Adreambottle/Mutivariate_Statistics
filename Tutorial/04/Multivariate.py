#!/usr/bin/env python
# coding: utf-8

# In[118]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from IPython.display import display, HTML
from scipy import stats

sns.set_style("whitegrid")


# In[119]:


# data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
data = pd.read_csv("wine.data")
data.columns = ["V"+str(i) for i in range(1, len(data.columns)+1)]  # rename column names to be similar to R naming convention
data.V1 = data.V1.astype(str)
X = data.loc[:, "V2":]  # independent variables data
y = data.V1  # dependednt variable data


# In[120]:


X.describe()


# In[121]:


y.describe()


# In[122]:


data_test = data.loc[:, "V2":"V6"]
print(data_test.head())


# In[123]:


pd.plotting.scatter_matrix(data_test, diagonal="kde")
plt.tight_layout()
plt.show()


# In[124]:


sns.lmplot("V4", "V5", data, hue="V1", fit_reg=False)


# In[125]:


ax = data[["V2","V3","V4","V5","V6"]].plot()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[126]:


X.apply(np.mean)


# In[127]:


X.apply(np.std)


# In[128]:


class2data = data[y=="2"]
class2data.loc[:, "V2":].apply(np.mean)


# In[129]:


class2data.loc[:, "V2":].apply(np.std)


# ### Between-groups Variance and Within-groups Variance for a Variable
# If we want to calculate the within-groups variance for a particular variable (for example, for a particular chemical’s concentration), we can use the function `calcWithinGroupsVariance()` below:

# In[130]:


def calcWithinGroupsVariance(variable, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    # get the mean and standard deviation for each group:
    numtotal = 0
    denomtotal = 0
    for leveli in levels:
        levelidata = variable[groupvariable==leveli]
        levelilength = len(levelidata)
        # get the standard deviation for group i:
        sdi = np.std(levelidata)
        numi = (levelilength)*sdi**2
        denomi = levelilength
        numtotal = numtotal + numi
        denomtotal = denomtotal + denomi
    # calculate the within-groups variance
    Vw = numtotal / (denomtotal - numlevels)
    return Vw

calcWithinGroupsVariance(X.V2, y)


# In[131]:


def calcBetweenGroupsCovariance(variable1, variable2, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    # calculate the grand means
    variable1mean = np.mean(variable1)
    variable2mean = np.mean(variable2)
    # calculate the between-groups covariance
    Covb = 0.0
    for leveli in levels:
        levelidata1 = variable1[groupvariable==leveli]
        levelidata2 = variable2[groupvariable==leveli]
        mean1 = np.mean(levelidata1)
        mean2 = np.mean(levelidata2)
        levelilength = len(levelidata1)
        term1 = (mean1 - variable1mean) * (mean2 - variable2mean) * levelilength
        Covb += term1
    Covb /= numlevels - 1
    return Covb

calcBetweenGroupsCovariance(X.V8, X.V11, y)


# ### Calculating Correlations for Multivariate Data
# It is often of interest to investigate whether any of the variables in a multivariate data set are significantly correlated.

# In[132]:


corr = stats.pearsonr(X.V2, X.V3)
print("p-value:\t", corr[1])
print("cor:\t\t", corr[0])


# In[133]:


corrmat = X.corr()
corrmat


# In[134]:


sns.heatmap(corrmat, vmax=1., square=False).xaxis.tick_top()


# ### Principal Component Analysis
# The purpose of principal component analysis is to find the best low-dimensional representation of the variation in a multivariate data set.

# If you want to compare different variables that have different units, are very different variances, it is a good idea to first standardise the variables.
# 
# Thus, it would be a better idea to first standardise the variables so that they all have variance 1 and mean 0, and to then carry out the principal component analysis on the standardised data. This would allow us to find the principal components that provide the best low-dimensional representation of the variation in the original data, without being overly biased by those variables that show the most variance in the original data.
# 
# To carry out a principal component analysis (PCA) on a multivariate data set, the first step is often to standardise the variables under study using the `scale()` function (see above).
# 
# This is necessary if the input variables have very different variances, which is true in this case as the concentrations of the 13 chemicals have very different variances (see above).
# 
# Once you have standardised your variables, you can carry out a principal component analysis using the `PCA` class from `sklearn.decomposition` package and its fit method, which fits the model with the data X. The default solver is Singular Value Decomposition (“svd”).
# 

# In[135]:


standardisedX = scale(X)
standardisedX = pd.DataFrame(standardisedX, index=X.index, columns=X.columns)


# In[136]:


standardisedX.apply(np.mean)


# In[137]:


standardisedX.apply(np.std)


# In[146]:


pca = PCA(n_components=1)
pca.fit(standardisedX)
pca.components_


# In[147]:


pca = PCA(n_components=2)
pca.fit(standardisedX)
pca.components_


# In[ ]:





# In[145]:


pca = PCA().fit(standardisedX)
# pca = PCA(n_components=1)
# pca.fit(standardisedX)
# pca.components_


# In[139]:


def pca_summary(pca, standardised_data, out=True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(zip(a, b, c), index=names, columns=columns)
    if out:
        print("Importance of components:")
        display(summary)
    return summary

summary = pca_summary(pca, standardisedX)


# In[140]:


def screeplot(pca, standardised_values):
    y = np.std(pca.transform(standardised_values), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()

screeplot(pca, standardisedX)

summary.sdev**2


# ### Loadings for the Principal Components
# The loadings for the principal components are stored in a named element `components_` of the variable returned by PCA().fit(). This contains a matrix with the loadings of each principal component, where the first column in the matrix contains the loadings for the first principal component, the second column contains the loadings for the second principal component, and so on.

# In[141]:


print(pca.components_[0])
np.sum(pca.components_[0]**2)


# This means that the first principal component is a linear combination of the variables:
# $$1 = \sum_{i=1}^{p}cp_iZ_i$$

# In[142]:


print(pca.components_[1])
np.sum(pca.components_[0]**2)


# The values of the principal components can be computed by the transform() (or fit_transform()) method of the PCA class. It returns a matrix with the principal components, where the first column in the matrix contains the first principal component, the second column the second component, and so on.
# 
# Thus, in our example, pca.transform(standardisedX)[:, 0] contains the first principal component, and pca.transform(standardisedX)[:, 1] contains the second principal component.
# 

# In[ ]:


def pca_scatter(pca, standardised_values, classifs):
    foo = pca.transform(standardised_values)
    bar = pd.DataFrame(zip(foo[:, 0], foo[:, 1], classifs), columns=["PC1", "PC2", "Class"])
    sns.lmplot("PC1", "PC2", bar, hue="Class", fit_reg=False)

pca_scatter(pca, standardisedX, y)


# ### Linear Discriminant Analysis
# The purpose of principal component analysis is to find the best low-dimensional representation of the variation in a multivariate data set. 
# 
# The purpose of linear discriminant analysis (LDA) is to find the linear combinations of the original variables that gives the best possible separation between the groups (wine cultivars here) in our data set. Linear discriminant analysis is also known as canonical discriminant analysis, or simply discriminant analysis.
# 

# In[143]:





# 

# In[143]:





# 

# In[ ]:




