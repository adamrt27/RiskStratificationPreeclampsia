import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample, seed
import math
import re
import dill as pickle

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn_extra.cluster import KMedoids
import hdbscan 
from sklearn.cluster import AffinityPropagation
from sklearn.feature_selection import chi2, f_oneway
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn import preprocessing
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, confusion_matrix,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from scipy.stats import shapiro, levene, bartlett, f_oneway, chi2_contingency

from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *

import gower
from itertools import chain
from scipy.sparse import issparse

import snf


class DataSet:
    
    """
    Represents a dataset containing various data and labels, and provides functionalities for data processing.

    Parameters:
        input_data (pd.DataFrame): The main input data as a pandas DataFrame.
        input_data_unscaled (pd.DataFrame): Unscaled version of the input data, if applicable. Used for plotting.
        bio_data (pd.DataFrame): Biomarker data as a pandas DataFrame.
        pe_labels (pd.DataFrame): Labels for treatment or intervention.
        site_labels (pd.DataFrame): Labels indicating different sites or locations.
        outcome_bin_cat (pd.DataFrame): Outcome data only including binary or categorical variables.
        outcome_cont_ord (pd.DataFrame): Outcome data only including continuous or ordinal variables.
        data_dict (pd.DataFrame): Data dictionary containing information about the variables.
        projection (pd.DataFrame): Projection of data, if applicable.
        gower (bool): Flag to compute Gower distance matrix.
        daisy (bool): Flag to use daisy for Gower distance computation. This involves using euclidian distance 
                      for continous variables
        split_input (bool): Flag to split input data into binary/categorical and continuous/ordinal subsets.
        empty (bool): Flag to create an empty dataset without any computations. Used when loading in a DataSet.
        SNF (bool): Flag to perform Similarity Network Fusion (SNF).

    Attributes:
        input_data (pd.DataFrame): The main input data as a pandas DataFrame.
        input_data_unscaled (pd.DataFrame): Unscaled version of the input data, if applicable.
        pe_labels (pd.DataFrame): Labels for treatment or intervention.
        site_labels (pd.DataFrame): Labels indicating different sites or locations.
        outcome_bin_cat (pd.DataFrame): Outcome data only including binary or categorical variables.
        outcome_cont_ord (pd.DataFrame): Outcome data only including continuous or ordinal variables.
        data_dict (pd.DataFrame): Data dictionary containing information about the variables.
        projection (pd.DataFrame): Projection of data, if applicable.
        labels (dict): A dictionary to store different clustering labels.
        bio_data (pd.DataFrame): Biomarker data as a pandas DataFrame.
        gower (pd.DataFrame): Gower distance matrix.
        snf_affinity (pd.DataFrame): Similarity Network Fusion (SNF) affinity matrix (0 is least related).
        snf_dist (pd.DataFrame): SNF distance matrix (0 is most related).
        input_bin_cat (pd.DataFrame): Subset of input data containing binary/categorical variables.
        input_cont_ord (pd.DataFrame): Subset of input data containing continuous/ordinal variables.
    """
    
    def __init__(self, input_data = pd.DataFrame(), input_data_unscaled = pd.DataFrame(), bio_data = pd.DataFrame(), 
                 pe_labels = pd.DataFrame(),site_labels = pd.DataFrame(), outcome_bin_cat = pd.DataFrame(), 
                 outcome_cont_ord = pd.DataFrame(), data_dict = pd.DataFrame(), projection = pd.DataFrame(),gower = True, 
                 daisy=True, split_input = True, empty = False, SNF = True):
        
        self.input_data = input_data
        self.input_data_unscaled = input_data_unscaled
        self.pe_labels = pe_labels
        self.site_labels = site_labels
        self.outcome_bin_cat = outcome_bin_cat
        self.outcome_cont_ord = outcome_cont_ord
        self.projection = projection
        self.labels = {"kMedoids": None, "HDBSCAN": None, "AffinityPropagation": None, "AgglomerativeClustering": None,
                      "SpectralClustering":None}
        self.bio_data = bio_data
        self.data_dict = data_dict
        
        if not empty:

            cols = self.input_data.columns
            all_cols = list(self.data_dict["Variable SAS name"])
            cat = []

            for i in range(len(all_cols)):
                if all_cols[i] in cols:
                    if ("ontinuous" not in list(self.data_dict["Database Categories"])[i]) and ("rdinal" not in list(self.data_dict["Database Categories"])[i]):
                        cat.append([all_cols[i],True])
                    else:
                        cat.append([all_cols[i],False])

            res = pd.DataFrame(cat)

            res = res.set_index(0).loc[self.input_data.columns].reset_index()

            cat_correct_order = list(res[1])
            
            if split_input:

                self.input_bin_cat, self.input_cont_ord = DataSet.subset_data(self.input_data, cat_correct_order)
                
            if gower:
                self.gower = self.get_gower_matrix(daisy=daisy, cat = cat_correct_order)
            else:
                self.gower = None
                
            if SNF:
                risk_data_dist = snf.compute.affinity_matrix(self.gower)
                bio_data_dist = snf.compute.make_affinity(bio_data)
                
                snf_matrix = pd.DataFrame(snf.compute.snf([bio_data_dist,risk_data_dist]))
                
                self.snf_affinity = snf_matrix
                self.snf_dist = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(snf_matrix.pow(-1)))
            else:
                self.snf_affinity = None
                self.snf_dist = None
                
        else:
            self.gower = None
            self.snf_dist = None
            self.snf_affinity = None
                
                


    # Saves the dataset for later use
    
    def save_DataSet(self,dir):
           
        """
        Save the DataSet object to a file.

        Args:
            dir (str): The directory path where the DataSet object will be saved.

        Returns:
            None
        """
            
        pickle.dump(self,open(dir, "wb"))
        
    def open_DataSet(self, dir):
        
        """
        Load a DataSet object from a file.

        Args:
            dir (str): The directory path from where the DataSet object will be loaded.

        Returns:
            DataSet: The loaded DataSet object.
        """
        
        data = pickle.load(open(dir, "rb"))
        
        self.input_data = data.input_data.copy()
        self.input_data_unscaled = data.input_data_unscaled.copy()
        self.pe_labels = data.pe_labels.copy()
        self.site_labels = data.site_labels.copy()
        self.outcome_bin_cat = data.outcome_bin_cat.copy()
        self.outcome_cont_ord = data.outcome_cont_ord.copy()
        self.data_dict = data.data_dict
        self.projection = data.projection
        self.labels = data.labels
        self.gower = data.gower.copy()
        self.snf_affinity = data.snf_affinity
        self.snf_dist = data.snf_affinity
        self.input_bin_cat = data.input_bin_cat
        self.input_cont_ord = data.input_cont_ord
        self.bio_data = data.bio_data
        
        return data
    
    ## Data Prep Functions
    
    def PrepForSemiSupervised(self, data, labels, fill_val = np.nan):
        
        """
        Prepares the data and labels for semi-supervised clustering by converting the provided labels to a list and
        filling any missing values with a specified fill value.

        Parameters:
            data (pd.DataFrame): The input data to prepare for semi-supervised clustering.
            labels (pd.Series): The class labels for the data.
            fill_val (float or int): The value to fill in for missing values in the labels. Default is np.nan.

        Returns:
            list: A list containing the prepared labels for semi-supervised clustering.
        """
        
        labels_semi = []

        temp = pd.DataFrame(labels,index= data.index)
        temp2 = pd.DataFrame(np.full((1,len(self.input_data.index)),fill_val).flatten(),index = self.input_data.index)
        temp2.update(temp)

        labels_semi.append(np.asarray(temp2).ravel())

        return labels_semi
    
    def subset_data(data, type):
       
        """
        Subsets the input data based on the provided boolean type list. Used to split input data into categorical 
        and continouos

        Parameters:
            data (pd.DataFrame): The input data to be subsetted.
            type (list of bool): A boolean list indicating whether to include each column (True) or exclude it (False).

        Returns:
            tuple: A tuple containing two DataFrames - the first DataFrame includes the columns specified by the True values
                   in the 'type' list, and the second DataFrame includes the columns specified by the False values in 
                   the 'type' list.
        """
        
        temp = pd.DataFrame()
        temp1 = pd.DataFrame()

        for i in range(len(data.columns)):

            if type[i]:
                temp = temp.copy()
                temp[data.columns[i]] = data[data.columns[i]] 
            else:
                temp1 = temp1.copy()
                temp1[data.columns[i]] = data[data.columns[i]] 

        return temp, temp1

    def gower_daisy_matrix(data_x, data_y=None, weight=None, cat_features=None):  

        """
        Computes the Gower's distance matrix between two datasets (data_x and data_y) based on their categorical and 
        numeric features.

        Gower's distance is a measure of similarity between data points that handles mixed data types (categorical and 
        numeric).
        The Gower's distance ranges between 0 and 1, where 0 indicates that the data points are identical and 1 indicates that
        the data points are completely dissimilar.

        Parameters:
            data_x (pd.DataFrame or np.ndarray): The first dataset (can be a DataFrame or a NumPy array).
            data_y (pd.DataFrame or np.ndarray, optional): The second dataset to compute the distance against data_x.
                                                          If not provided, the function computes the distance between 
                                                          data_x and itself (default is None).
            weight (np.ndarray, optional): An array containing weights for each feature in the datasets.
                                          It should have the same length as the number of columns in data_x and data_y.
                                          If not provided, all features are assumed to have equal weights. (default is None).
            cat_features (array-like, optional): A boolean array or list indicating the categorical features in the datasets.
                                                 True indicates a categorical feature, and False indicates a numeric feature.
                                                 
                                                 If not provided, the function automatically identifies the 
                                                 categorical features.
                                                 (default is None).

        Returns:
            np.ndarray: A 2D array representing the Gower's distance matrix between data_x and data_y.

        Raises:
            TypeError: If the input data_x and data_y are sparse matrices, or if they have different shapes/columns.
        """
        
        # function checks
        X = data_x
        if data_y is None: Y = data_x 
        else: Y = data_y 
        if not isinstance(X, np.ndarray): 
            if not np.array_equal(X.columns, Y.columns): raise TypeError("X and Y must have same columns!")   
        else: 
             if not X.shape[1] == Y.shape[1]: raise TypeError("X and Y must have same y-dim!")  

        if issparse(X) or issparse(Y): raise TypeError("Sparse matrices are not supported!")        

        x_n_rows, x_n_cols = X.shape
        y_n_rows, y_n_cols = Y.shape 

        if cat_features is None:
            if not isinstance(X, np.ndarray): 
                is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
                cat_features = is_number(X.dtypes)    
            else:
                cat_features = np.zeros(x_n_cols, dtype=bool)
                for col in range(x_n_cols):
                    if not np.issubdtype(type(X[0, col]), np.number):
                        cat_features[col]=True
        else:          
            cat_features = np.array(cat_features)

        # print(cat_features)

        if not isinstance(X, np.ndarray): X = np.asarray(X)
        if not isinstance(Y, np.ndarray): Y = np.asarray(Y)

        Z = np.concatenate((X,Y))

        x_index = range(0,x_n_rows)
        y_index = range(x_n_rows,x_n_rows+y_n_rows)

        Z_num = Z[:,np.logical_not(cat_features)]

        num_cols = Z_num.shape[1]
        num_ranges = np.zeros(num_cols)
        num_max = np.zeros(num_cols)

        for col in range(num_cols):
            col_array = Z_num[:, col].astype(np.float32) 
            max = np.nanmax(col_array)
            min = np.nanmin(col_array)

            if np.isnan(max):
                max = 0.0
            if np.isnan(min):
                min = 0.0
            num_max[col] = max
            num_ranges[col] = np.abs(1 - min / max) if (max != 0) else 0.0

        # This is to normalize the numeric values between 0 and 1.
        Z_num = np.divide(Z_num ,num_max,out=np.zeros_like(Z_num), where=num_max!=0)
        Z_cat = Z[:,cat_features]

        if weight is None:
            weight = np.ones(Z.shape[1])

        #print(weight)    

        weight_cat=weight[cat_features]
        weight_num=weight[np.logical_not(cat_features)]   

        out = np.zeros((x_n_rows, y_n_rows), dtype=np.float32)

        weight_sum = weight.sum()

        X_cat = Z_cat[x_index,]
        X_num = Z_num[x_index,]
        Y_cat = Z_cat[y_index,]
        Y_num = Z_num[y_index,]

       # print(X_cat,X_num,Y_cat,Y_num)

        for i in range(x_n_rows):          
            j_start= i        
            if x_n_rows != y_n_rows:
                j_start = 0
            # call the main function

            res = DataSet.gower_daisy_get(X_cat[i,:], 
                              X_num[i,:],
                              Y_cat[j_start:y_n_rows,:],
                              Y_num[j_start:y_n_rows,:],
                              weight_cat,
                              weight_num,
                              weight_sum,
                              cat_features,
                              num_ranges,
                              num_max) 
            #print(res)
            out[i,j_start:]=res
            if x_n_rows == y_n_rows: out[i:,j_start]=res

        return out

    def gower_daisy_get(xi_cat,xi_num,xj_cat,xj_num,feature_weight_cat,
                  feature_weight_num,feature_weight_sum,categorical_features,
                  ranges_of_numeric,max_of_numeric ):

        """
        Calculate Gower's distance between two data points using the daisy approach.

        Gower's distance is a similarity measure that can handle mixed data types (categorical and numerical).
        The daisy approach is used to compute the Euclidean distances between continuous (numerical) features.

        Parameters:
            xi_cat (numpy array): Categorical features of data point xi.
            xi_num (numpy array): Numerical features of data point xi.
            xj_cat (numpy array): Categorical features of data point xj.
            xj_num (numpy array): Numerical features of data point xj.
            feature_weight_cat (numpy array): Weights for categorical features.
            feature_weight_num (numpy array): Weights for numerical features.
            feature_weight_sum (float): Sum of all feature weights.
            categorical_features (numpy array): Boolean array indicating which features are categorical (True) and 
                                                which are numerical (False).
            ranges_of_numeric (numpy array): Array containing the ranges of numerical features.
            max_of_numeric (numpy array): Array containing the maximum values of numerical features.

        Returns:
            np.ndarray: Gower's distance between the two data points (xi and xj). The value is between 
                                   0 and 1, where 0 indicates that the data points are identical, and 1 indicates 
                                   they are completely dissimilar.
        """
        
        # categorical columns
        sij_cat = np.where(xi_cat == xj_cat,np.zeros_like(xi_cat),np.ones_like(xi_cat))
        sum_cat = np.multiply(feature_weight_cat,sij_cat).sum(axis=1) 

        # numerical columns
        sij_num=np.square(xi_num-xj_num)
        sum_num = np.sqrt(np.multiply(feature_weight_num,sij_num).sum(axis=1))

        sums= np.add(sum_cat,sum_num)
        sum_sij = np.divide(sums,feature_weight_sum)

        return sum_sij

    def get_gower_matrix(self, data = pd.DataFrame(), daisy = True, cat = []):

        """
        Compute the Gower's distance matrix for a given dataset using either the daisy or the gower package.

        Gower's distance is a measure of similarity between data points that handles mixed data types (categorical and 
        numeric). The Gower's distance matrix contains pairwise distances between all data points, where each entry (i, j)
        represents the similarity between data point i and data point j.

        Parameters:
            data (pd.DataFrame, optional): The dataset for which to compute the Gower's distance matrix.
                                           If not provided, the function uses the input_data attribute of the DataSet class.
                                           (default is pd.DataFrame())
            daisy (bool, optional): A boolean flag to indicate whether to use the daisy approach or the gower package
                                    to compute Gower's distance matrix. If True, the daisy approach is used;
                                    if False, the gower package is used. (default is True)

        Returns:
            pd.DataFrame: A square DataFrame representing the Gower's distance matrix. The row and column labels are the 
                          index values of the input data. Also sets self.gower to this DataFrame.
        """
        
        set_object = True
        
        if len(data) == 0:
            data = self.input_data
            set_object=False
        
        # get all categorical columns

        if len(cat) == 0:
            cols = data.columns
            all_cols = list(self.data_dict["Variable SAS name"])
            cat = []

            for i in range(len(all_cols)):
                if all_cols[i] in cols:
                    if ("ontinuous" not in list(self.data_dict["Database Categories"])[i]) and ("rdinal" not in list(self.data_dict["Database Categories"])[i]):
                        cat.append(True)
                    else:
                        cat.append(False)
            
        # Make gower matrix
        
        data_gower = np.asarray(data)
        
        if daisy:
        
            data_gower = DataSet.gower_daisy_matrix(data_gower, cat_features = cat)
            
        else:
            
            data_gower = gower.gower_matrix(data_gower, cat_features = cat)
        
        df_gower  = pd.DataFrame(data_gower,index = data.index, columns = data.index)
        
        
        if set_object:
            self.gower = df_gower
        
        return df_gower
    
    def flatten(l):
        """
        This function takes a list of lists and returns a flattened list containing all the elements from the nested lists.

        Parameters:
            l (list of lists): The input list of lists to be flattened.

        Returns:
            list: A flattened list containing all the elements from the nested lists.
        """
        
        return [item for sublist in l for item in sublist]
    
    def propensity_matcher(self, caliper = 0.1):
    
        """
        Perform propensity score matching to balance covariates between treated and control groups.

        Propensity score matching is a statistical method used to reduce bias in observational studies when comparing the
        effects of a treatment (in this case, 'PE' treatment) on an outcome. The method identifies control subjects that are
        similar to treated subjects based on their propensity scores, which are the predicted probabilities of receiving the
        treatment.

        Parameters:
            caliper (float, optional): The maximum allowable distance between matched pairs. A smaller caliper leads to closer
                                       matches between treated and control subjects, but may reduce the number of 
                                       matched pairs. (default is 0.1).

        Returns:
            pd.DataFrame: A new DataFrame containing the matched samples from the original input_data, with covariates
                          balanced between treated and control groups.
        """
    
        ds = self.input_data.copy()
    
        ds["patient"] = self.input_data.index
        ds["PE"] = np.asarray(self.pe_labels)
        
        # Generating Matches
        
        psm = PsmPy(ds, treatment = 'PE', indx='patient', exclude = [])
        psm.logistic_ps(balance = False)
        psm.predicted_data
        psm.knn_matched(matcher='propensity_logit',caliper= caliper)
        matched_rep = psm.matched_ids
        
        # Combining all PE and matched patients into a list
        
        p = [matched_rep["patient"].tolist()]    
        
        p.append(matched_rep["matched_ID"].tolist())
        
        p = self.flatten(p)
        
        # Putting those patients from list into a dataframe

        data_matched_rep = self.input_data.copy()
        for id in data_matched_rep["patient"]:
            if id not in p:
                data_matched_rep.drop([id],axis=0,inplace=True)

        data_matched_rep.set_index(keys="patient",inplace=True)
        
        return data_matched_rep
    
    
   
    
    ## Clustering Functions
    
    def num_clusters(self, method = "both", algorithm = 'kMedoids'):
    
        """
        Determine the optimal number of clusters using the elbow and silhouette methods.

        Args:
            method (str): The method to use for determining the number of clusters. Options are 'elbow', 
                          'silhouette', or 'both'.
            algorithm (str): The clustering algorithm to use for determining the number of clusters. Options 
                             are 'kMedoids' or 'KMeans'.

        Returns:
            matplotlib.figure.Figure or tuple: If method is 'elbow', returns the elbow plot figure. 
            
                                               If method is 'silhouette', returns a tuple containing silhouette 
                                               plot figure and a dictionary of silhouette scores for different 
                                               cluster sizes. 
                                               
                                               If method is 'both', returns a tuple containing both elbow and 
                                               silhouette plot figures along with the silhouette scores dictionary.
        """
    
        if method == "both":
        
            fig = plt.figure(figsize=(16,8))
            
        else:
            
            fig = plt.figure(figsize=(8,8))
            
        
        # Elbow
        
        if method in ['elbow','both']:
            
            distortions = []
            
            K = range(1,20)
            
            for k in K:
                
                if algorithm == 'kMedoids':

                    kmeanModel = KMedoids(n_clusters=k)
                    
                else:
                    
                    kmeanModel = KMeans(n_clusters=k)
                    
                kmeanModel.fit(self.input_data)
                distortions.append(kmeanModel.inertia_)
            
            ax = fig.add_subplot(1,2,1)
            ax.plot(K, distortions, 'bx-')
            
        # Silhoutte
        
        if method in ["silhouette",'both']:   

            range_n_clusters = [2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15]

            X = np.asarray(self.input_data)

            silhouette_avgs = {}

            for n_clusters in range_n_clusters:

                if algorithm == 'kMedoids':
                    clusterer = KMedoids(n_clusters=n_clusters, random_state=10)
                else:
                    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                    
                cluster_labels = clusterer.fit_predict(X)


                silhouette_avg = silhouette_score(X, cluster_labels)

                silhouette_avgs[n_clusters] = silhouette_avg


            ax2 = fig.add_subplot(1,2,2)
            ax2.plot(list(silhouette_avgs.keys()),list(silhouette_avgs.values()))

            return fig,silhouette_avgs
        return fig
    
    
    def kMedoids_func(self,data = pd.DataFrame(),n_clusters = 2,bootstrap = False, n_boot = 10, boot_ratio = 0.2, seed = 42, **kwargs):
        
        """
        Perform k-Medoids clustering on the data.

        K-Medoids clustering is a variant of K-Means that uses medoids (data points that are representative of clusters)
        instead of centroids. It is particularly useful when dealing with non-Euclidean distances or when the data contains
        outliers.

        Parameters:
            data (pd.DataFrame, optional): The input data for clustering (can be a DataFrame or a NumPy array).
                                           If not provided, the function uses the DataSet's input_data attribute as the
                                           input data (default is pd.DataFrame()).
            n_clusters (int, optional): The number of clusters to form (default is 2).
            bootstrap (bool, optional): Whether to perform bootstrap clustering (default is False).
                                        If True, the function performs bootstrap clustering and returns a list of arrays
                                        containing cluster labels for each bootstrap iteration.
                                        If False, the function performs a single k-Medoids clustering and returns the
                                        KMedoids clustering result object (and a figure with the clustered data plotted,
                                        if projection is available).
            n_boot (int, optional): The number of bootstrap iterations to perform if bootstrap is True (default is 10).
            boot_ratio (float, optional): The ratio of data points to include in each bootstrap sample, if bootstrap is True
                                          (default is 0.2).
            seed (int, optional): The random seed for reproducibility, used when generating bootstrap samples (default is 42).
            **kwargs: Additional keyword arguments to be passed to the KMedoids constructor.

        Returns:
            list or tuple: If bootstrap is False, returns either the KMedoids clustering result object (and a figure with
                           the clustered data plotted, if projection is available).
                           If bootstrap is True, returns a list of arrays containing cluster labels for each bootstrap
                           iteration.
        """
        
        if not bootstrap:
            if len(data) == 0:
                res = KMedoids(n_clusters=n_clusters,**kwargs).fit(self.input_data)
            else:
                res = KMedoids(n_clusters=n_clusters,**kwargs).fit(data)

            self.labels["kMedoids"] = res.labels_

            labels = res.labels_

            if len(self.projection):
                fig = plt.figure(figsize=(15,15))

                plt.scatter(*self.projection.T,c=labels,alpha=0.5)
                plt.title("K-Medoids {}".format(str(self.input_data)))

                return fig, res

            else:
                return res
        else:
            
            res_labels = []
            
            for i in range(n_boot):
                
                if len(data) == 0:
                    data = self.input_data
                    
                sample,spread = train_test_split(data, test_size = boot_ratio, random_state=seed + i)
                
                res = KMedoids(n_clusters=n_clusters,**kwargs).fit(sample)
                
                labels = self.PrepForSemiSupervised(sample, res.labels_, fill_val = -1)
                
                label_propagation_model = LabelPropagation(kernel='knn', n_neighbors=10)

                # Fit the model with the labeled data
                label_propagation_model.fit(data, np.ravel(labels))
                
                y_pred = label_propagation_model.predict(data)
                
                res_labels.append(y_pred)
                                
            return res_labels
        

    def generate_colors(color = 'deep', cluster = None):
        
        """
        Generate colors for visualizing clustered data points.

        This function generates a color palette based on a specified color map and assigns colors to each data point based on
        their cluster labels.

        Parameters:
            color (str, optional): The name of the color palette to use (default is 'deep').
                                   The color palette should be a valid Seaborn palette name (e.g., 'deep', 'muted', 'pastel',
                                   'bright', etc.).
            cluster (KMedoids clustering result object, optional): The clustering result object obtained from k-Medoids
                                                                  clustering.
                                                                  If not provided, the function will not generate colors
                                                                  based on cluster labels (default is None).

        Returns:
            list: A list of RGB tuples representing the colors assigned to each data point.
                  If the cluster parameter is None, the function
        """
        
        color_palette = sns.color_palette(color, cluster.labels_.max()+1)
        cluster_colors = [color_palette[x] if x >= 0
                        else (0.5, 0.5, 0.5)
                        for x in cluster.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                zip(cluster_colors, cluster.probabilities_)]
        
        return cluster_member_colors


    def HDBSCAN_func(self,data = pd.DataFrame(),precomputed=False, bootstrap = False, n_boot = 10, boot_ratio = 0.2, seed = 42, color_palette = 'deep',**kwargs):

        """
        Perform HDBSCAN clustering on the data.

        Parameters:
            data (pd.DataFrame, optional): The dataset on which to perform clustering.
                                           If not provided, the function uses the input_data of the DataSet object.
            precomputed (bool, optional): Whether the distance matrix is precomputed (default is False).
                                          If True, the input data is assumed to be a distance matrix rather than raw data.
            bootstrap (bool, optional): Whether to perform bootstrap clustering (default is False).
                                        If True, the function performs bootstrapping with multiple iterations of HDBSCAN.
                                        The number of iterations is determined by n_boot.
            n_boot (int, optional): The number of bootstrapping iterations (default is 10).
            boot_ratio (float, optional): The ratio of data to include in each bootstrap sample (default is 0.2).
            seed (int, optional): The seed for the random number generator used in bootstrapping (default is 42).
            color_palette (str, optional): The name of the color palette to use for visualizing clusters (default is 'deep').
                                           The color palette should be a valid Seaborn palette name (e.g., 'deep', 'muted',
                                           'pastel', 'bright', etc.).
            **kwargs: Additional keyword arguments to pass to the HDBSCAN algorithm.

        Returns:
            hdbscan.HDBSCAN or list: If bootstrap is False, returns the HDBSCAN clustering result object (and a figure with
                                     the clustered data plotted, if projection is available).
                                     If bootstrap is True, returns a list of arrays containing cluster labels for each
                                     bootstrap iteration.
        """
        
        if not bootstrap:
            if len(data) == 0:
                if precomputed:
                    res = hdbscan.HDBSCAN(**kwargs,metric='precomputed').fit(np.asarray(self.gower,dtype="double"))
                else:
                    res = hdbscan.HDBSCAN(**kwargs).fit(np.asarray(self.input_data,dtype="double"))

            else:
                if precomputed:
                    res = hdbscan.HDBSCAN(**kwargs,metric='precomputed').fit(data)
                else:
                    res = hdbscan.HDBSCAN(**kwargs).fit(data)

            self.labels["HDBSCAN"] = res.labels_

            cluster_member_colors = DataSet.generate_colors(color = color_palette, cluster = res)

            if len(self.projection) > 0:
                fig = plt.figure(figsize=(15,15))

                ax = fig.add_subplot(1,1,1)

                ax.scatter(*self.projection.T,c=cluster_member_colors,alpha = 0.5,cmap='plasma')
                ax.set_title("HDBSCAN {}".format(str(self.input_data)))

                return fig, res

            return res
        else:
            res_labels = []
            
            for i in range(n_boot):
                
                if len(data) == 0:
                    data = self.input_data
                    
                sample,spread = train_test_split(data, test_size = boot_ratio, random_state=seed + i)
                
                if precomputed:
                    res = hdbscan.HDBSCAN(**kwargs,metric='precomputed').fit(np.asarray(self.get_gower_matrix(data=sample),dtype="double"))
                else:
                    res = hdbscan.HDBSCAN(**kwargs).fit(np.asarray(sample,dtype="double"))
                
                labels = self.PrepForSemiSupervised(sample, res.labels_, fill_val = -2)
                
                label_propagation_model = LabelPropagation(kernel='knn', n_neighbors=10)
                
                labels = np.asarray(labels) + 1
                                
                # Fit the model with the labeled data
                label_propagation_model.fit(data, np.ravel(labels))
                
                y_pred = label_propagation_model.predict(data)
                
                y_pred = np.asarray(y_pred) - 1
                
                res_labels.append(y_pred)
                                
            return res_labels



    def AffinityPropagation_func(self, data = pd.DataFrame(), precomputed=False, bootstrap = False, n_boot = 10, boot_ratio = 0.2, seed = 42, color_palette='deep', **kwargs):
        
        """
        Perform Affinity Propagation clustering on the data.

        Args:
            data (pd.DataFrame, optional): The dataset on which to perform clustering.
                                           If not provided, the function uses the input_data of the DataSet object.
            precomputed (bool, optional): Whether the similarity matrix is precomputed (default is False).
                                          If True, the input data is assumed to be a similarity matrix rather than raw data.
            bootstrap (bool, optional): Whether to perform bootstrap clustering (default is False).
                                        If True, the function performs bootstrapping with multiple iterations of 
                                        Affinity Propagation. The number of iterations is determined by n_boot.
            n_boot (int, optional): The number of bootstrapping iterations (default is 10).
            boot_ratio (float, optional): The ratio of data to include in each bootstrap sample (default is 0.2).
            seed (int, optional): The seed for the random number generator used in bootstrapping (default is 42).
            color_palette (str, optional): The name of the color palette to use for visualizing clusters (default is 'deep').
                                           The color palette should be a valid Seaborn palette name (e.g., 'deep', 'muted',
                                           'pastel', 'bright', etc.).
            **kwargs: Additional keyword arguments to pass to the AffinityPropagation algorithm.

        Returns:
            AffinityPropagation or list: If bootstrap is False, returns the Affinity Propagation clustering result object 
                                         (and a figure with the clustered data plotted, if projection is available).
                                         If bootstrap is True, returns a list of arrays containing cluster labels for each
                                         bootstrap iteration.
        """
        
        if not bootstrap:
            if len(data) == 0:
                if precomputed:
                    res = AffinityPropagation(affinity='precomputed',**kwargs).fit(self.gower)
                else:
                    res = AffinityPropagation(**kwargs).fit(self.input_data)
            else:
                if precomputed:
                    res = AffinityPropagation(affinity='precomputed',**kwargs).fit(data)
                else:
                    res = AffinityPropagation(**kwargs).fit(data)

            self.labels["AffinityPropagation"] = res.labels_

            num_labels = res.labels_.max()
            color_palette = sns.color_palette(color_palette, num_labels+2)
            cluster_colors = [color_palette[x] if x >= 0
                            else (0.5, 0.5, 0.5)
                            for x in res.labels_]

            if len(self.projection) > 0:
                fig = plt.figure(figsize=(15,15))
                plt.scatter(*self.projection.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)
                plt.title("Affinity Propagation {}".format(str(self.input_data)))

                return fig, res
            return res
        else:
            res_labels = []
            
            for i in range(n_boot):
                
                if len(data) == 0:
                    data = self.input_data
                    
                sample,spread = train_test_split(data, test_size = boot_ratio, random_state=seed + i)
                
                if precomputed:
                    res = AffinityPropagation(**kwargs,affinity='precomputed').fit(self.get_gower_matrix(data=sample))
                else:
                    res = AffinityPropagation(**kwargs).fit(sample)
                
                labels = self.PrepForSemiSupervised(sample, res.labels_, fill_val = -1)
                
                label_propagation_model = LabelPropagation(kernel='knn', n_neighbors=10)

                # Fit the model with the labeled data
                label_propagation_model.fit(data, np.ravel(labels))
                
                y_pred = label_propagation_model.predict(data)
                
                res_labels.append(y_pred)
                                
            return res_labels
    

    def AgglomerativeClustering_func(self, data = pd.DataFrame(), n_clusters = 2,linkage = "average",precomputed = False, bootstrap = False, n_boot = 10, boot_ratio = 0.2, seed = 42, color_palette = 'deep',**kwargs):
        
        """
        Perform Agglomerative Clustering on the data.

        Args:
            data (pd.DataFrame, optional): The dataset on which to perform clustering.
                                           If not provided, the function uses the input_data of the DataSet object.
            n_clusters (int, optional): The number of clusters to form (default is 2).
            precomputed (bool, optional): Whether the similarity matrix is precomputed (default is False).
                                          If True, the input data is assumed to be a similarity matrix rather than raw data.
            bootstrap (bool, optional): Whether to perform bootstrap clustering (default is False).
                                        If True, the function performs bootstrapping with multiple iterations of 
                                        Agglomerative Clustering.
                                        The number of iterations is determined by n_boot.
            n_boot (int, optional): The number of bootstrapping iterations (default is 10).
            boot_ratio (float, optional): The ratio of data to include in each bootstrap sample (default is 0.2).
            seed (int, optional): The seed for the random number generator used in bootstrapping (default is 42).
            color_palette (str, optional): The name of the color palette to use for visualizing clusters (default is 'deep').
                                           The color palette should be a valid Seaborn palette name (e.g., 'deep', 'muted',
                                           'pastel', 'bright', etc.).
            **kwargs: Additional keyword arguments to pass to the AgglomerativeClustering algorithm.

        Returns:
            AgglomerativeClustering or list: If bootstrap is False, returns the Agglomerative Clustering result object (and 
                                             a figure with the clustered data plotted, if projection is available).
                                             If bootstrap is True, returns a list of arrays containing cluster labels for each
                                             bootstrap iteration.
        """
        
        if not bootstrap:
            if len(data) == 0:
                if precomputed:
                    res = AgglomerativeClustering(n_clusters =n_clusters,
                                                  affinity='precomputed',
                                                  linkage = linkage,
                                                  **kwargs).fit(self.gower)
                else:
                    res = AgglomerativeClustering(n_clusters = n_clusters,
                                                  linkage = linkage,**kwargs).fit(self.input_data)
            else:
                if precomputed:
                    res = AgglomerativeClustering(n_clusters = n_clusters,
                                                  linkage = linkage,affinity='precomputed',**kwargs).fit(data)
                else:
                    res = AgglomerativeClustering(n_clusters = n_clusters,
                                                  linkage = linkage,**kwargs).fit(data)

            self.labels["AgglomerativeClustering"] = res.labels_

            num_labels = res.labels_.max()
            color_palette = sns.color_palette(color_palette, num_labels+2)
            cluster_colors = [color_palette[x] if x >= 0
                            else (0.5, 0.5, 0.5)
                            for x in res.labels_]

            if len(self.projection) > 0:
                fig = plt.figure(figsize=(15,15))
                plt.scatter(*self.projection.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)
                plt.title("Agglomerative Clustering {}".format(str(self.input_data)))

                return fig, res
            return res
        else:
            res_labels = []
            
            for i in range(n_boot):
                
                if len(data) == 0:
                    data = self.input_data
                    
                sample,spread = train_test_split(data, test_size = boot_ratio, random_state=seed + i)
                
                if precomputed:
                    res = AgglomerativeClustering(n_clusters = n_clusters,
                                                  affinity='precomputed',
                                                  linkage = linkage,
                                                  **kwargs).fit(self.get_gower_matrix(data=sample))
                else:
                    res = AgglomerativeClustering(n_clusters = n_clusters,
                                                  linkage = linkage,
                                                  **kwargs).fit(sample)
                
                labels = self.PrepForSemiSupervised(sample, res.labels_, fill_val = -1)
                
                label_propagation_model = LabelPropagation(kernel='knn', n_neighbors=10)

                # Fit the model with the labeled data
                label_propagation_model.fit(data, np.ravel(labels))
                
                y_pred = label_propagation_model.predict(data)
                
                res_labels.append(y_pred)
                                
            return res_labels
    
    
    def SpectralClustering_func(self, data = pd.DataFrame(), n_clusters = 2,precomputed = False, bootstrap = False, n_boot = 10, boot_ratio = 0.2, seed = 42, color_palette = "deep",**kwargs):
        
        """
        Perform Spectral Clustering on the data.

        Args:
            data (pd.DataFrame, optional): The dataset on which to perform clustering.
                                           If not provided, the function uses the input_data of the DataSet object.
            n_clusters (int, optional): The number of clusters to form (default is 2).
            precomputed (bool, optional): Whether the similarity matrix is precomputed (default is False).
                                          If True, the input data is assumed to be a similarity matrix rather than raw data.
            bootstrap (bool, optional): Whether to perform bootstrap clustering (default is False).
                                        If True, the function performs bootstrapping with multiple iterations of 
                                        Spectral Clustering.
                                        The number of iterations is determined by n_boot.
            n_boot (int, optional): The number of bootstrapping iterations (default is 10).
            boot_ratio (float, optional): The ratio of data to include in each bootstrap sample (default is 0.2).
            seed (int, optional): The seed for the random number generator used in bootstrapping (default is 42).
            color_palette (str, optional): The name of the color palette to use for visualizing clusters (default is 'deep').
                                           The color palette should be a valid Seaborn palette name (e.g., 'deep', 'muted',
                                           'pastel', 'bright', etc.).
            **kwargs: Additional keyword arguments to pass to the SpectralClustering algorithm.

        Returns:
            SpectralClustering or list: If bootstrap is False, returns the Spectral Clustering result object (and a 
                                        figure with the clustered data plotted, if projection is available) 
                                        If bootstrap is True, returns a list of arrays containing cluster labels for each
                                        bootstrap iteration.
        """
        
        if not bootstrap:
            if len(data) == 0:
                if precomputed:
                    res = SpectralClustering(n_clusters = n_clusters,affinity='precomputed', random_state = 42, **kwargs).fit(self.snf_affinity)
                else:
                    res = SpectralClustering(n_clusters = n_clusters, random_state = 42, **kwargs).fit(self.input_data)
            else:
                if precomputed:
                    res = SpectralClustering(n_clusters = n_clusters, random_state = 42, affinity='precomputed',**kwargs).fit(data)
                else:
                    res = SpectralClustering(n_clusters = n_clusters,random_state = 42,**kwargs).fit(data)

            self.labels["SpectralClustering"] = res.labels_

            num_labels = res.labels_.max()
            color_palette = sns.color_palette(color_palette, num_labels+2)
            cluster_colors = [color_palette[x] if x >= 0
                            else (0.5, 0.5, 0.5)
                            for x in res.labels_]

            if len(self.projection) > 0:
                fig = plt.figure(figsize=(15,15))
                plt.scatter(*self.projection.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)
                plt.title("Agglomerative Clustering {}".format(str(self.input_data)))

                return fig, res

            return res

        else:
            res_labels = []
            
            for i in range(n_boot):
                
                if len(data) == 0:
                    data = self.input_data
                    
                sample,spread = train_test_split(data, test_size = boot_ratio, random_state=seed + i)
                
                if precomputed:
                    res = SpectralClustering(n_clusters = n_clusters,affinity='precomputed',**kwargs).fit(self.get_gower_matrix(data=sample))
                else:
                    res = SpectralClustering(n_clusters = n_clusters,**kwargs).fit(sample)
                
                labels = self.PrepForSemiSupervised(sample, res.labels_, fill_val = -1)
                
                label_propagation_model = LabelPropagation(kernel='knn', n_neighbors=10)

                # Fit the model with the labeled data
                label_propagation_model.fit(data, np.ravel(labels))
                
                y_pred = label_propagation_model.predict(data)
                
                res_labels.append(y_pred)
                                
            return res_labels
        
    ## Statistics Functions
    
    # Performs ANOVA assuming columns contain continous dependant variables 
    # and the last columns has the class labels, called "labels". Outputs
    # a dataframe with the fstat and pvalue for each variable.

    def ANOVA(self, labels, subset = [], data = "output", range_type = None, range_level = 0.9,covar = "site", output_type = "full"):
        
        """
        Perform ANOVA statistical testing between clusters for the given clustering solution.

        Args:
            labels (array): Labels corresponding to cluster assignments.
            subset (list, optional): List of indices or labels to subset the data (default is an empty list).
            data (str, optional): A string specifying the type of data to use for testing. Possible values: "output", "input", "bio" (default is "output").
            range_type (str, optional): Type of range to consider for outlier removal. Possible values: "quantile", "confidence", or None (default is None).
            range_level (float, optional): Level of the range for outlier removal (default is 0.9).
            covar (str, optional): Covariate variable for ANCOVA (default is "site").
            output_type (str, optional): Type of output to return. Possible values: "full", "clean" (default is "full").

        Returns:
            pandas.DataFrame: A DataFrame containing ANOVA results for each variable.

        Notes:
            This function performs ANOVA statistical testing between clusters based on the provided labels. It supports different data types ("output", "input", "bio"),
            outlier removal using quantile or confidence interval range, and covariate-based ANCOVA.

            The function calculates various ANOVA-related statistics, including p-values, effect sizes, Welch's F-test, and Shapiro-Wilk normality tests.
            The output DataFrame contains results for each variable, including statistics, p-values, and validity flags based on normality assumptions.

            The output type can be set to "full" to include all statistics or "clean" to include only valid cases with respect to normality assumptions.
        """
        
        import pingouin
        
        if len(subset) == 0:
            subset = list(self.input_data.index)
        
        if data == "output":
            df = self.outcome_cont_ord.copy().loc[subset]
        elif data == "input":
            df = self.input_cont_ord.copy().loc[subset]
        elif data == "bio":
            df = self.bio_data.copy()

            mean_values = df[~np.isinf(df)].mean()

            # Step 2: Replace infinite values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Step 3: Fill NaN values with the calculated mean
            df.fillna(mean_values, inplace=True)

        df["labels"] = labels
        df["site"] = DataSet.str_list_to_int(np.ravel(self.site_labels.copy().loc[subset]))
        
        ANOVA = []
        Welch_p = []
        Welch_f = []
        Welch_effect = []
        ancova_p = []
        ancova_f = []
        ancova_effect = []
        
        s = []
        l = []
        b = []
        valid = []
        
        effect = []
        
        n_labels = int(df["labels"].max() - df["labels"].min() + 1)
        

        for var in df:
            
            if var not in ['labels',"PE","site"]:
                
                if range == "quantile":
                    
                    margin = (1 - range_level) / 2
                    inner_lower_quantile = margin
                    inner_upper_quantile = 1 - margin
                    
                    q1 = df[col].quantile(inner_lower_quantile)
                    q3 = df[col].quantile(inner_upper_quantile)
                    iqr = q3 - q1

                    # Calculate the lower and upper bounds for the IQR range
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    # Select data within the IQR range
                    iqr_data = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
                    df_temp1 = iqr_data.copy()
                    
                if range == "confidence":
                    
                    # Calculate the mean and standard error of the selected column
                    mean = df[col].mean()
                    std_error = df[col].std() / np.sqrt(len(df))

                    # Define the confidence level (e.g., 95%)

                    # Compute the margin of error
                    margin_of_error = stats.t.ppf((1 + range_level) / 2, df=len(df)-1) * std_error

                    # Calculate the lower and upper bounds of the confidence interval
                    lower_bound = mean - margin_of_error
                    upper_bound = mean + margin_of_error

                    # Subset the data to values within the confidence interval
                    subset_data = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    
                    df_temp1 = subset_data.copy()
                    
                else:
                    df_temp1 = df.copy()
                    
                
                # Drop NA and create temp df
                
                df_temp = df_temp1.dropna(how="any",subset=[var]).copy()
                
                # check there are enough variables
                
                if len(df_temp[var]) >= 3:
                
                    # Test for normal Distrobution

                    s.append(shapiro(df_temp[var])[1])
                    p = shapiro(df_temp[var])[1]

                    if p < 0.05:

                        # Testing for ANOVA use: Levene

                        p_1 = levene(df_temp[var],df_temp["labels"])[1]
                        p_2 = None

                        if p_1 >= 0.05:
                            valid.append(True)

                        else:
                            valid.append(False)

                    else:

                        # Testing for ANOVA use: Bartlett

                        p_1 = None
                        p_2 = bartlett(df_temp[var],df_temp["labels"])[1]

                        if p_2 >= 0.05:
                            valid.append(True)

                        else:
                            valid.append(False)

                    l.append(p_1)
                    b.append(p_2)    

                    samples = []

                    for i in range(n_labels):
                        if len(df[df["labels"] == i][var].dropna().tolist()) > 0:
                            samples.append(df[df["labels"] == i][var].dropna())

                    if len(samples) > 1:
                        ANOVA.append(f_oneway(*samples))
                    else:
                        ANOVA.append([None, None])
                
                    #print(var)
                
                    Welch_p.append(pingouin.welch_anova(df_temp,between="labels",dv=var)["p-unc"][0])
                    Welch_f.append(pingouin.welch_anova(df_temp,between="labels",dv=var)["F"][0])
                    Welch_effect.append(pingouin.welch_anova(df_temp,between="labels",dv=var)["np2"][0])
#                     ancova_p.append(pingouin.ancova(df_temp, covar = covar, between="labels",dv=var)["p-unc"][0])
#                     ancova_f.append(pingouin.ancova(df_temp, covar = covar, between="labels",dv=var)["F"][0])
#                     ancova_effect.append(pingouin.ancova(df_temp, covar = covar, between="labels",dv=var)["np2"][0])
                    effect.append(DataSet.correlation_ratio(df, labels, var))
                
                else:
                    
                    s.append(np.nan)
                    valid.append(np.nan)
                    ANOVA.append(np.nan)
                    ancova.append(np.nan)
                    Welch_f.append(np.nan)
                    Welch_p.append(np.nan)
                    Welch_effect.append(np.nan)
                    l.append(np.nan)
                    b.append(np.nan)
                    effect.append(np.nan)
                

        if data == "output":
            res = pd.DataFrame(ANOVA,index = df.columns[0:-2], columns = ["statistic","pvalue"])
        else:
            res = pd.DataFrame(ANOVA,index = df.columns[0:-2], columns = ["statistic","pvalue"])        
        
        res["effect"] = effect
        res["welch statistic"] = Welch_f
        res["welch pvalue"] = Welch_p
        res["welch effect"] = Welch_effect
#         res["ANCOVA statistic"] = ancova_f
#         res["ANCOVA pvalue"] = ancova_p
#         res["ANCOVA effect"] = ancova_effect
        #res["effect size"] = effect
        res["shapiro"] = s
        res["levene"] = l
        res["bartlett"] = b
        res["valid"] = valid
             
        if output_type == "full":
            return res
        elif output_type == "clean":
            res1 = pd.DataFrame(index = res.index, columns = ["statistic","pvalue", "effect size"])

            for ind in res1.index:
                if res.loc[ind, "valid"] == True:
                    res1.loc[ind] = list(res.loc[ind, ["statistic","pvalue", "effect"]])
                else:
                    res1.loc[ind] = list(res.loc[ind, ["welch statistic","welch pvalue", "welch effect"]])
            
            return res1
            
            
    def str_list_to_int(l):

        """
        Convert a list of strings into a list of corresponding integer values.

        Args:
            l (list): A list of strings to be converted to integers.

        Returns:
            list: A list of integer values corresponding to the input strings.
        """
        
        idx = {}
        for x in l:
            if x not in idx:
                idx[x] = len(idx)

        return [idx[x] for x in l]    
        
    # Performs CHI2 test assuming columns contain binary or categorical dependant
    # variables and the last column has the class labels, called "labels". Outputs
    # a dataframe with the fstat and pvalue for each variable.

    def CHI2(self,labels,data = "output", subset = [], range_type = None, range_level = 0.9):
        
        """
        Perform Chi-squared statistical testing between categorical variables and cluster assignments.

        Args:
            labels (array): Labels corresponding to cluster assignments.

            data (str, optional): A string indicating which data to use for the analysis ('output', 'input', or 'bio').
                                  If not provided, the function uses the output binary categorical data of the DataSet object.

            subset (list, optional): A list of indices representing the subset of data to consider (default is an empty list).

            range_type (str, optional): The type of range to consider for filtering data, either 'quantile' or 'confidence'
                                        (default is None).

            range_level (float, optional): The confidence level or quantile level for the range (default is 0.9).
                                          Only applicable when range_type is specified.

        Returns:
            pandas.DataFrame: A DataFrame containing Chi-squared statistics, p-values, and effect sizes
                              for each categorical variable with respect to cluster assignments.
        """
        
        if len(subset) == 0:
            subset = list(self.input_data.index)
        
        if data == "output":
            df = self.outcome_bin_cat.copy().loc[subset]
        elif data == "input":
            df = self.input_bin_cat.copy().loc[subset]
        elif data == "bio":
            return None

        df["labels"] = labels

        
        CHI2 = []

        for col in df:
            if col != 'labels':
                
                if range == "quantile":
                    
                    margin = (1 - range_level) / 2
                    inner_lower_quantile = margin
                    inner_upper_quantile = 1 - margin
                    
                    q1 = df[col].quantile(inner_lower_quantile)
                    q3 = df[col].quantile(inner_upper_quantile)
                    iqr = q3 - q1

                    # Calculate the lower and upper bounds for the IQR range
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    # Select data within the IQR range
                    iqr_data = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
                    cont_table = pd.crosstab(index=iqr_data["labels"],columns=iqr_data[col]).T
                    
                    cram = DataSet.cramers_v(iqr_data, iqr_data["labels"], col)
                    
                if range == "confidence":
                    
                    # Calculate the mean and standard error of the selected column
                    mean = df[col].mean()
                    std_error = df[col].std() / np.sqrt(len(df))

                    # Define the confidence level (e.g., 95%)

                    # Compute the margin of error
                    margin_of_error = stats.t.ppf((1 + range_level) / 2, df=len(df)-1) * std_error

                    # Calculate the lower and upper bounds of the confidence interval
                    lower_bound = mean - margin_of_error
                    upper_bound = mean + margin_of_error

                    # Subset the data to values within the confidence interval
                    subset_data = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

                    cont_table = pd.crosstab(index=subset_data["labels"],columns=subset_data[col]).T
                    
                    cram = DataSet.cramers_v(subset_data, subset_data["labels"], col)
                else:
                    cont_table = pd.crosstab(index=df["labels"],columns=df[col]).T
                    
                    try:
                    
                        cram = DataSet.cramers_v(df, labels, col)
                        
                    except:
                        
                        cram = np.nan
                    
                try:
                    stat, pval, expected_freq = chi2_contingency(cont_table)[0], chi2_contingency(cont_table)[1], chi2_contingency(cont_table)[3].tolist()[0]
                    
                except:
                    
                    stat, pval, expected_freq = np.nan, np.nan, np.nan
                    
                CHI2.append([stat,pval, cram])
                
        return pd.DataFrame(CHI2,columns=['statistic','pvalue', "effect size"],index=df.columns[0:-1])


    # Get Histogram

    def get_histogram(data,labels,var, seperate = True, n_clust = None):
        
        """
        Generate histograms for a variable based on cluster assignments.

        Args:
            data (pandas.DataFrame): The data containing the variables.
            labels (array): Labels corresponding to cluster assignments.
            var (str): The name of the variable to plot.
            separate (bool, optional): Whether to generate separate histograms for each cluster (default is True).
            n_clust (int, optional): The number of clusters to consider. If None, determined from maximum label value.

        Returns:
            matplotlib.figure.Figure: A Figure containing histograms of the specified variable based on cluster assignments.
        """
        
        if n_clust == None:
            n_clust = int(max(labels)) + 1

        cont_list = data.copy()
        cont_list["labels"] = labels

        num_lab = int(max(labels) + 1)

        if seperate:
            fig, ax = plt.subplots(n_clust,1,figsize=(10,10), sharex="all")

        else:
            fig = plt.figure(figsize=(10,10))

            
        x = [cont_list.dropna(how="any",subset=var)[cont_list.dropna(how="any",subset=var).labels == i][var] for i in range(num_lab)]

        lab = [i for i in range(len(x))]

        if not seperate:
            plt.hist(x,label = lab)
            plt.legend()
        
        else:
            for i in range(n_clust):
                ax[i].hist(x[i],label = lab[i], bins = 20)
                ax[i].title.set_text(lab[i])

        plt.title('Histogram of {}'.format(var))
        return fig



    # Run an random forest interpretable classifier to identify important variables.
    # Data is the variables you want to know the importance of, and labels is the cluster labels
    # returns a pandas DataFrame with variables ranked by importance, and prints out the accuracy
    # of the classifier.

    def random_forest_importance(self,data = pd.DataFrame(),labels = pd.DataFrame(), n_class = 2):
        
        """
        Calculate feature importances and evaluate a random forest classifier.

        Args:
            data (pandas.DataFrame, optional): The input data for training and testing the classifier (default is an empty DataFrame).
            labels (pandas.DataFrame, optional): The corresponding labels for the input data (default is an empty DataFrame).
            n_class (int, optional): The number of classes for classification (default is 2).

        Returns:
            pandas.DataFrame: A DataFrame containing feature importances and other evaluation metrics of the random forest classifier.
        """
        
        # Split data into testing and training
        
        if len(data) == 0:
            data = self.input_data
            
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)
            
        # Build and fit random forest

        clf = RandomForestClassifier(n_estimators=1000,verbose=0,criterion="entropy")
        clf = clf.fit(X_train, y_train)

        # Encode categorical variables for accuracy testing

        le = preprocessing.LabelEncoder()
        y_test_enc = le.fit_transform(y_test)

        # Get predicted variables for AUC

        y_pred =  le.fit_transform(clf.predict(X_test))

        # Get accuracy score

        print("ACC:",clf.score(X_test,y_test))

        # Get balanced accuracy score
        
    

        print("Bal ACC:",balanced_accuracy_score(y_test_enc,y_pred))

        # Get AUC Score
        
        if n_class > 2:
            
            try:
                print("AUC:",roc_auc_score(y_test_enc, clf.predict_proba(X_test),multi_class='ovr'))
            except ValueError as e:
                print(f"error: {e}")
        
        else:
            
            print("AUC:",roc_auc_score(y_test_enc, clf.predict(X_test)))

        print(confusion_matrix(y_test,clf.predict(X_test)))

        # Build Dataframe with most batch effected features

        features = pd.DataFrame()
        features["variables"] = data.columns
        features["importances"] = clf.feature_importances_
        features = features.sort_values("importances",ascending=False)
        
        return features


    # Do everything given a set of labels
    
    def EvaluateClusters(self,labels = None, data = "output" , subset = [], range_type = None, range_level = 0.9, output_type = "clean"):
        
        """
        Evaluate clustering results using CHI2 and ANOVA statistical tests.

        Args:
            labels (array-like): Labels corresponding to cluster assignments.
            data (str, optional): The type of data to be used for evaluation ('output', 'input', or 'bio', default is 'output').
            subset (list, optional): Subset of data indices to be used for evaluation (default is an empty list).
            range_type (str, optional): The type of range to be used for data filtering ('quantile', 'confidence', or None, default is None).
            range_level (float, optional): The level of the range (default is 0.9).
            output_type (str, optional): Type of output from ANOVA ('clean' or 'full', default is 'clean').

        Returns:
            tuple: A tuple containing DataFrames of CHI2 and ANOVA results if applicable, otherwise returns (None, None).
            
            The returned CHI2 DataFrame contains statistical results from the Chi-squared (CHI2) test, which measures the association
            between categorical variables and cluster assignments. It includes columns like 'statistic', 'pvalue', and 'effect size',
            which quantify the significance and strength of the association. 

            The returned ANOVA DataFrame contains statistical results from the Analysis of Variance (ANOVA) test, which measures the
            differences in means of continuous or ordinal variables across clusters. It includes columns like 'statistic', 'pvalue',
            'effect size', 'welch statistic', 'welch pvalue', and more. The 'output_type' parameter determines whether the DataFrame
            contains the complete ANOVA statistics or a simplified version with only the relevant information.
        """
        
        labels = DataSet.remove_outliers(labels)
        
        if len(np.unique(labels)) == 1 and np.unique(labels)[0] == -1:
            return None
        
        if data == "output":
            data_bin_cat = self.outcome_bin_cat.copy().loc[subset]
            data_cont_ord = self.outcome_cont_ord.copy().loc[subset]
        elif data == "input":
            data_bin_cat = self.input_bin_cat.copy().loc[subset]
            data_cont_ord = self.input_cont_ord.copy().loc[subset]
        elif data == "bio":
            df = self.bio_data.copy()

            mean_values = df[~np.isinf(df)].mean()

            # Step 2: Replace infinite values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Step 3: Fill NaN values with the calculated mean
            df.fillna(mean_values, inplace=True)
            
            data_bin_cat = pd.DataFrame()
            data_cont_ord = df.loc[subset]
                
        if not data_bin_cat.equals(pd.DataFrame()):
            chi_df = self.CHI2(labels, data = data, subset = subset,range_type = range_type, range_level = range_level)
        else:
            chi_df = None
        
        
        if not data_cont_ord.equals(pd.DataFrame()):
            anova_df = self.ANOVA(labels, data = data, subset=subset, range_type = range_type, range_level = range_level, output_type= output_type)
        else:
            anova_df = None
        
        return chi_df, anova_df
        
    def get_cluster_sum_var(data, labels, var):
        
        """
        Calculate the mean and median of a variable within each cluster.

        Args:
            data (pd.DataFrame): DataFrame containing the variables.
            labels (array-like): Labels corresponding to cluster assignments.
            var (str): Variable for which to calculate the statistics. This corresponds to a column in data.

        Returns:
            list: A list of lists, where each sublist contains the mean and median
                  of the specified variable within a cluster.

        This function calculates and returns the mean and median values of the specified
        variable for each cluster present in the provided labels. The data is organized
        based on the cluster labels, and the mean and median values are calculated for
        each cluster. The result is returned as a list of lists, where each sublist contains
        the mean and median values for a particular cluster.
        """
        
        res = []
        
        n_clust = int(max(np.ravel(labels)) + 1)

        cont_list = data.copy()
        cont_list["labels"] = np.ravel(labels)

        x = [cont_list.dropna(how="any",subset=var)[cont_list.dropna(how="any",subset=var).labels == i][var] for i in range(n_clust)]

        for i in range(n_clust):
            #print("cluster_{} mean =".format(i),x[i].mean()," median =", x[i].median())
            res.append([x[i].mean(),x[i].median()])
            
        return res

    # Removes outliers from labels

    def remove_outliers(labels = None, outlier_val = -1):

        """
        Remove outlier values from a list of labels.

        Args:
            labels (array-like): List of labels containing outlier values.
            outlier_val (int or float, optional): Value to be treated as an outlier (default is -1).

        Returns:
            list: A list of labels with outlier values replaced by NaN.

        This function takes a list of labels and replaces outlier values (specified by
        outlier_val) with NaN (Not a Number). It returns a new list with outlier values
        replaced by NaN, while keeping other values intact.
        """
        
        labels = list(labels)

        for i in range(len(labels)):
            if labels[i] == outlier_val:
                labels[i] = np.nan

        return labels
        

    def get_val_counts(data, labels, var):

        """
        Get value counts of a variable within clusters.

        Args:
            data (DataFrame): Data containing the variable and labels.
            labels (array-like): Cluster labels corresponding to data.
            var (str): Name of the variable for which to compute value counts. Corresponds to a column in data.

        Returns:
            list of list of dict: A nested list of dictionaries, where each sublist corresponds to a cluster.
                                  Each dictionary in the sublist contains value-count pairs for the variable.

        This function computes the value counts of a specified variable within each cluster defined by labels.
        It returns a nested list where each sublist corresponds to a cluster, and each dictionary in the sublist
        contains value-count pairs for the variable. The dictionaries have keys as variable values and values as
        their respective counts within the cluster.
        """
        
        res = []
        
        data = data.copy()
        
        data["Labels"] = labels
        for i in range(int(max(labels) + 1)):
           # print("cluster", i)
            lol = [{v:c} for v,c in data[data["Labels"] == i][var].value_counts().items()]
            #print(lol)
            res.append(lol)
            
        original_array = res
      # Determine the maximum number of categories
        max_category = int(max(max(subdict.keys(), default=0) for sublist in original_array for subdict in sublist))

        # Transform the original array
        transformed_array = [
            [subdict.get(i, 0) for i in range(1, max_category + 1)]
            for sublist in original_array
            for subdict in sublist
        ]

        # Reshape the transformed array to have the same number of sublists
        num_sublists = len(original_array)
        transformed_array = [transformed_array[i:i+num_sublists] for i in range(0, len(transformed_array), num_sublists)]

        return res

    def add(m,n):
        if m == None:
            return n
        elif n == None:
            return m
        return m+n

    def correlation_ratio(df, labels,col):
        
        """
        Compute the correlation ratio (Eta) between a categorical variable and cluster labels.

        Args:
            df (DataFrame): Data containing the variable, labels, and cluster assignments.
            labels (array-like): Cluster labels corresponding to data.
            col (str): Name of the categorical variable for which to compute the correlation ratio. Should be a column in data.

        Returns:
            float: The computed correlation ratio (Eta) between the categorical variable and cluster labels.

        This function calculates the correlation ratio (Eta) between a categorical variable and cluster labels.
        The correlation ratio measures the proportion of the total variance in the variable that is explained by
        the differences between clusters. It ranges from 0 to 1, where 0 indicates no association, and 1 indicates
        a perfect association between the variable and the cluster labels.
        """
        
        df = df.copy()
        df["labels"] = labels
        
        categories = np.asarray(df.dropna(how="any",subset=col).labels)
        measurements = np.asarray(df.dropna(how="any",subset=col)[col])
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat)+1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0,cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
        numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
        denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = np.sqrt(numerator/denominator)
        return eta


    def cramers_v(df, labels,col):
        
        """
        Calculate the Cramer's V statistic between a categorical variable and cluster labels.

        Args:
            df (DataFrame): Data containing the variable, labels, and cluster assignments.
            labels (array-like): Cluster labels corresponding to data.
            col (str): Name of the categorical variable for which to compute Cramer's V. Should be a column in data.

        Returns:
            float: The computed Cramer's V statistic between the categorical variable and cluster labels.

        This function calculates the Cramer's V statistic, which measures the strength of association between
        a categorical variable and cluster labels. It is an extension of the chi-squared statistic, taking into
        account the dimensions of the contingency table. Cramer's V ranges from 0 to 1, where 0 indicates no
        association, and 1 indicates a strong association between the variable and the cluster labels.
        """
        
        df = df.copy()
        df["labels"] = labels
        
        x = np.asarray(df.dropna(how="any",subset=col)[col])
        y = x = np.asarray(df.dropna(how="any",subset=col)["labels"])
        confusion_matrix = pd.crosstab(x,y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
        rcorr = r-((r-1)**2)/(n-1)
        kcorr = k-((k-1)**2)/(n-1)
        return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
    
    def compute_medoids_from_distances(self,labels,data = pd.DataFrame(), distances = pd.DataFrame()):
        
        """
        Compute medoids for each cluster based on distance matrix.

        Args:
            labels (array-like): Cluster labels corresponding to data points.
            data (DataFrame, optional): Input data. If not provided, uses the input_data of the object.
            distances (DataFrame, optional): Distance matrix. If not provided, uses the Gower distance matrix of the object.

        Returns:
            DataFrame: A DataFrame containing the computed medoids for each cluster.

        This function calculates medoids for each cluster based on a distance matrix. Medoids are data points within
        clusters that have the minimum total distance to all other data points in the same cluster. It is a more robust
        measure than centroids when dealing with non-Euclidean distance metrics.
        """
        
        if data.empty:
            data_arr = np.asarray(self.input_data.copy())
            data = self.input_data.copy()
        else:
            data_arr = np.asarray(data)
            
        if distances.empty:
            distances = np.asarray(self.gower.copy())
        else:
            distances = np.asarray(distances)

        
        
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        medoids = []

        for label in unique_labels:
            cluster_data = data_arr[labels == label]

            # Extract distances within the cluster
            cluster_distances = distances[labels == label][:, labels == label]

            # Calculate the total distance from each point to others
            total_distances = np.sum(cluster_distances, axis=1)

            # Find the index of the data point with the minimum total distance (medoid)
            medoid_idx = np.argmin(total_distances)

            # Append the medoid to the list
            medoids.append(cluster_data[medoid_idx])

        return pd.DataFrame(np.array(medoids), columns=data.columns, index=["medoid_{}".format(i) for i in range(len(unique_labels))])
    
class MatchedDataSet(DataSet):
    
    """
    A class for creating a matched dataset using the parent DataSet.

    Parameters:
        DataSet (DataSet, optional): The parent DataSet object userd in matching.
        ratio (float, optional): The ratio of matching samples to original samples (default is 1).
        n_selections (int, optional): The number of times to perform sample selection (default is 10).
        random_seed (int, optional): The random seed for reproducibility (default is 42).
        gower (bool, optional): Whether to use Gower distance for matching (default is True).
        empty (bool, optional): Whether to create an empty instance (default is False).
        SNF (bool, optional): Whether to use Similarity Network Fusion (SNF) for matching (default is True).
        
    Attributes:
        raw_data (DataSet): The parent DataSet object used in matching. 
        matched_data (array of DataSet objects): Array of matched DataSet objects.
    """
    
    def __init__(self, DataSet= None, ratio = 1, n_selections = 10, random_seed = 42, gower = True, empty = False, SNF = True):

        if empty:
            self.raw_data = None
            self.matched_data = None
        
        else:
            self.raw_data = DataSet
            self.matched_data = self.matcher(ratio, n_selections, random_seed, gower, SNF)
      
     # Saves the dataset for later use
    
    def save_MatchedDataSet(self,dir):
            
        """
        Save the MatchedDataSet object to a file using the pickle module.

        Args:
            dir (str): The directory path along with the filename to save the object.
                       Example: "/path/to/save/matched_dataset.pkl"
        """
            
        pickle.dump(self,open(dir, "wb"))
        
    def open_MatchedDataSet(self, dir):
        
        """
        Open a saved MatchedDataSet object from a file using the pickle module.

        Args:
            dir (str): The directory path along with the filename of the saved object.
                       Example: "/path/to/saved/matched_dataset.pkl"

        Returns:
            MatchedDataSet: The loaded MatchedDataSet object.
        """
        
        data = pickle.load(open(dir, "rb"))
        
        self.raw_data = data.raw_data
        
        self.matched_data = data.matched_data
        
        return data
    
        
    def PrepForSemiSupervised_matched(ds, labels, fill_val = np.nan):

        """
        Prepare labels for semi-supervised learning using a MatchedDataSet.

        This function takes a MatchedDataSet object and corresponding labels, and prepares the labels for
        semi-supervised learning by matching the labels with the original input_data indices. It fills in any
        missing indices with a specified fill value.

        Args:
            ds (MatchedDataSet): The MatchedDataSet object containing raw_data and matched_data.
            labels (list of arrays): A list of arrays containing cluster labels for each matched dataset.
                                     Each array should have the same length as the corresponding matched dataset.
            fill_val (float or other, optional): The value used to fill missing indices in the labels (default is np.nan).

        Returns:
            list of arrays: A list of arrays with labels prepared for semi-supervised learning.
                            Each array corresponds to a matched dataset's input_data indices.
        """
        
        labels_semi = []

        for i in range(len(labels)):
            temp = pd.DataFrame(labels[i],index= ds.matched_data[i].input_data.index)
            temp2 = pd.DataFrame(np.full((1,len(ds.raw_data.input_data.index)),fill_val).flatten(),index = ds.raw_data.input_data.index)
            temp2.update(temp)

            labels_semi.append(np.asarray(temp2).ravel())

        return labels_semi

    # Takes in data, a column to match on, the ratio (ie number of controls per PE patient), 
    # and the number of randomly generated dataframes (n_selections), and a random seed parameter 
    # and returns a list of Dataframes

    def matcher(self, ratio = 1, n_selections = 10, random_seed = 42, gower = True, SNF = True):
        
        """
        Generate matched datasets using the MatchedDataSet class.

        This function generates a list of matched datasets by creating new datasets with matched samples
        based on the specified ratio and number of selections. Each matched dataset consists of a set of control
        samples and a corresponding number of positive examples, aiming to balance the classes.

        Args:
            ratio (float, optional): The ratio of control samples to positive examples in the matched datasets (default is 1).
            n_selections (int, optional): The number of matched datasets to generate (default is 10).
            random_seed (int, optional): The seed for the random number generator used in sampling (default is 42).
            gower (bool, optional): Whether to compute Gower distance matrices for the matched datasets (default is True).
            SNF (bool, optional): Whether to apply spectral neighbor fusion (SNF) to the matched datasets (default is True).

        Returns:
            list of MatchedDataSet: A list of MatchedDataSet objects containing the generated matched datasets.
        """
        
        # Creating empty list to add matched data
        matched_list = []
        
        # Creating each dataframe
        for i in range(n_selections):
            
            res = []
            
            for ds in [self.raw_data.input_data.copy(), self.raw_data.bio_data.copy(), self.raw_data.outcome_bin_cat.copy(), self.raw_data.outcome_cont_ord.copy()]:
            
                data_temp = ds.copy()


                data_temp["PE"] = np.asarray(self.raw_data.pe_labels)
                data_temp["site"] = np.asarray(self.raw_data.site_labels)

                PE_temp = data_temp.loc[data_temp["PE"] == 1]
                control_temp = data_temp.loc[data_temp["PE"] == 0]

                control_temp = control_temp.sample(n= ratio * len(PE_temp["PE"].tolist()), axis=0, random_state = random_seed + i)

                res.append(pd.concat([control_temp,PE_temp], axis=0))
            
            res_dataset = DataSet(input_data = res[0].drop(columns = ["PE","site"]).copy(),
                                  bio_data = res[1].drop(columns=["PE","site"]).copy(),
                                  pe_labels = res[0]["PE"], site_labels = res[0]["site"], 
                                  outcome_bin_cat = res[2].drop(columns = ["PE","site"]).copy(), 
                                  outcome_cont_ord = res[3].drop(columns = ["PE","site"]).copy(), 
                                 data_dict = self.raw_data.data_dict.copy(), gower = gower, split_input = False, SNF = SNF)
            
            #res_dataset.gower = res_dataset.get_gower_matrix()  # (data = res[0].drop(columns = ["PE","site"]))
            
            matched_list.append(res_dataset)
            
        return matched_list