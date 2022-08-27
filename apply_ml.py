import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, ShuffleSplit

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from funcs import *
from plot_funcs import *


LASSO_DATA = None
PCA_DATA = None

def apply_KMeans(df, n_clusters=9):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df)
    return kmeans

def split_data_to_arrays(trainX, testX, trainY, testY):
    trainXarr = trainX.to_numpy() if not type(trainX) == np.ndarray else trainX
    trainYarr = trainY.to_numpy() if not type(trainY) == np.ndarray else trainY
    testYarr = testY.to_numpy() if not type(testY) == np.ndarray else testY
    testXarr = testX.to_numpy() if not type(testX) == np.ndarray else testX
    return trainXarr, testXarr, trainYarr, testYarr

def tune_KNN(df, labels, test_size=0.2):
    trainX, testX, trainY, testY = train_test_split(df, labels, test_size = test_size)
    trainXarr, testXarr, trainYarr, testYarr = split_data_to_arrays(trainX, testX, trainY, testY)
    knn = KNeighborsClassifier(n_jobs=-1)
    parameter_space = {
        'n_neighbors': [3,4,5,6],
        'algorithm': ['ball_tree', 'kd_tree'],
        'p':[1,2]
        }
    search = GridSearchCV(knn, parameter_space, n_jobs=-1, cv=5, scoring='accuracy')
    search.fit(trainXarr, trainYarr)
    #search.best_params_
    accuracy = search.best_score_
    y_pred = search.predict(testXarr)
    return search.best_estimator_, accuracy, classification_report(testYarr, y_pred, output_dict=True),\
        confusion_matrix(testYarr, y_pred), search.best_params_

def tune_SVM(df, labels, test_size=0.2):
    print("Start Tune SVM")
    trainX, testX, trainY, testY = train_test_split(df, labels, test_size = test_size)
    trainXarr, testXarr, trainYarr, testYarr = split_data_to_arrays(trainX, testX, trainY, testY)
    parameter_space = {
        'C': [1, 0.9, 0.75],
        'kernel': ['poly', 'rbf'],
        'degree':[3,4],
        'gamma': list(np.logspace(-3, 2, num=6))
        }
    svc = SVC(verbose=True)
    search = GridSearchCV(svc, parameter_space, n_jobs=-1, cv=5, scoring='accuracy')
    search.fit(trainXarr, trainYarr)
    #search.best_params_
    accuracy = search.best_score_
    best_clf = search.best_estimator_
    y_pred = best_clf.predict(testXarr)
    return search.best_estimator_, accuracy, classification_report(testYarr, y_pred, output_dict=True),\
        confusion_matrix(testYarr, y_pred), search.best_params_

def tune_LSVM(df, labels, test_size=0.2):
    print("Start Tune SVM")
    trainX, testX, trainY, testY = train_test_split(df, labels, test_size = test_size)
    trainXarr, testXarr, trainYarr, testYarr = split_data_to_arrays(trainX, testX, trainY, testY)
    parameter_space = {
        'penalty':['l1','l2'],
        'C': [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        'multi_class':['ovr', 'crammer_singer'],
        }
    svc = LinearSVC(verbose=True, max_iter=100000)
    search = GridSearchCV(svc, parameter_space, n_jobs=-1, cv=5, scoring='accuracy')
    search.fit(trainXarr, trainYarr)
    #search.best_params_
    accuracy = search.best_score_
    best_clf = search.best_estimator_
    y_pred = best_clf.predict(testXarr)
    return search.best_estimator_, accuracy, classification_report(testYarr, y_pred, output_dict=True),\
        confusion_matrix(testYarr, y_pred), search.best_params_

def tune_NaiveBayes(df, labels, test_size=0.2):
    trainX, testX, trainY, testY = train_test_split(df, labels, test_size = test_size)
    trainXarr, testXarr, trainYarr, testYarr = split_data_to_arrays(trainX, testX, trainY, testY)
    nb_clf = GaussianNB()
    cv = ShuffleSplit(n_splits=5, test_size=test_size)
    scores = cross_val_score(nb_clf, trainXarr, trainYarr, cv=cv, scoring='accuracy')
    nb_clf.fit(trainXarr, trainYarr)
    #test_acc = accuracy_score(testYarr, nb_clf.predict(testXarr))
    print('Accuracy of Gaussian Naive Bayes classifier after 5-fold cross validation: ', scores.mean())
    y_pred = nb_clf.predict(testXarr)
    return nb_clf, scores.mean(), classification_report(testYarr, y_pred, output_dict=True), confusion_matrix(testYarr, y_pred)

def tune_DecisionTree(df, labels, test_size=0.2):
    """Hardcoded for trying only the options of the 'criterion' parameter"""
    acc_tol = 1e-5
    trainX, testX, trainY, testY = train_test_split(df, labels, test_size = test_size)
    trainXarr, testXarr, trainYarr, testYarr = split_data_to_arrays(trainX, testX, trainY, testY)
    best_clfs = []
    param_vals = ['gini', 'entropy', 'log_loss']
    for parameter in param_vals:
        tree_clf = DecisionTreeClassifier(criterion=parameter, random_state=42)
        # Obtain the effective alphas in increasing order
        alphas_path = tree_clf.cost_complexity_pruning_path(trainXarr, trainYarr)
        alphas = alphas_path['ccp_alphas']
        # Obtain test accuracy of each alpha (fix random state for consistency)
        trees, accuracies = [], []
        for alpha in alphas:
            tree = DecisionTreeClassifier(criterion=parameter, random_state=42, ccp_alpha=alpha)
            tree.fit(trainXarr, trainYarr)
            trees.append(tree)
            accuracies.append(accuracy_score(testYarr, tree.predict(testXarr)))
        accuracies = np.array(accuracies)
        max_acc = accuracies.max()
        best_ind = sorted(np.where(accuracies >= max_acc-acc_tol)[0])[-1]
        best_acc, best_alpha = accuracies[best_ind], alphas[best_ind]
        tree = trees[best_ind]
        best_clfs.append({'clf':tree, 'acc':best_acc, 'alpha':best_alpha})
    best_clf = 0
    best_acc = 0
    for i, clf in enumerate(best_clfs):
        if clf['acc'] > best_acc:
            best_acc = clf['acc']
            best_clf = i
    best_clf = best_clfs[best_clf]['clf']
    # Cross validation for best metric estimation
    cv = ShuffleSplit(n_splits=5, test_size=test_size)
    scores = cross_val_score(best_clf, trainXarr, trainYarr, cv=cv, scoring='accuracy')
    cross_val_acc = scores.mean()
    best_clf.fit(trainXarr, trainYarr)
    y_pred = best_clf.predict(testXarr)
    return best_clf, cross_val_acc, classification_report(testYarr, y_pred, output_dict=True), confusion_matrix(testYarr, y_pred)

def tune_RandomForest(df, labels, test_size=0.2, best_alpha=0.001):
    max_trees = 200
    oob_tol = 1e-3
    trainX, testX, trainY, testY = train_test_split(df, labels, test_size = test_size)
    trainXarr, testXarr, trainYarr, testYarr = split_data_to_arrays(trainX, testX, trainY, testY)
    models, oobs, accuracies_rf = [], [], []
    # Train with different forest sizes, from 1 tree to max_trees
    for n in range(1, max_trees+1):
        rf_clf = RandomForestClassifier(n_estimators=n, bootstrap=True,
                                        ccp_alpha=best_alpha, oob_score=True,
                                        random_state=0)
        rf_clf.fit(trainXarr, trainYarr)
        cv = ShuffleSplit(n_splits=5, test_size=test_size)
        scores = cross_val_score(rf_clf, trainXarr, trainYarr, cv=cv, scoring='accuracy')
        models.append(rf_clf)
        oobs.append(rf_clf.oob_score_)
        accuracies_rf.append(scores.mean())
        # Stop when Relative Standard Deviation of the last 10 OOB accuracy
        # is under threshold (OOB is stable)
        if len(oobs)>10 and np.std(oobs[-10:]) / np.mean(oobs[-10:]) <= oob_tol:
            break
    best_acc_ix = np.array(accuracies_rf).argmax()
    best_clf = models[best_acc_ix]
    # Cross validation for best metric estimation
    cv = ShuffleSplit(n_splits=5, test_size=test_size)
    scores = cross_val_score(best_clf, trainXarr, trainYarr, cv=cv, scoring='accuracy')
    best_clf.fit(trainXarr, trainYarr)
    y_pred = best_clf.predict(testXarr)
    cross_val_acc = scores.mean()
    return best_clf, cross_val_acc, classification_report(testYarr, y_pred, output_dict=True), confusion_matrix(testYarr, y_pred)

def tune_MLP(df_in, labels, parameter_space=None, base_mlp=None, test_size=0.2):
    df = StandardScaler().fit(df_in).transform(df_in)
    trainX, testX, trainY, testY = train_test_split(df, labels, test_size = test_size)
    trainXarr, testXarr, trainYarr, testYarr = split_data_to_arrays(trainX, testX, trainY, testY)

    mlp = MLPClassifier(max_iter=2000) if not base_mlp else base_mlp
    if not parameter_space:
        parameter_space = {
        'hidden_layer_sizes': [(50,25,10), (45,90,20), (90,10),(45,90,90,45)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.01, 0.05],
        'learning_rate': ['constant','adaptive'],
        'early_stopping': [True, False]
        }

    search = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5, scoring='accuracy')
    search.fit(trainXarr, trainYarr)
    print('Best parameters found:\n', search.best_params_)
    #y_pred_cv = search.predict(testXarr)
    accuracy = search.best_score_
    best_clf = search.best_estimator_
    best_clf.fit(trainXarr, trainYarr)
    y_pred = best_clf.predict(testXarr)
    return best_clf, accuracy, classification_report(testYarr, y_pred, output_dict=True),\
        confusion_matrix(testYarr, y_pred), search.best_params_

def MLP_on_selected_features(df, labels, features=None):
    if features:
        df = df.loc[:,features]
    return tune_MLP(df, labels)

def MLP_on_lasso_features(df, labels):
    df = get_lasso_reduced_data(df, labels)
    return MLP_on_selected_features(df, labels)

def MLP_on_pca_features(df, labels, n_components=None, pca_explain=None):
    df = get_pca_reduced_data(df, labels, n_components=n_components, pca_explain=pca_explain)
    return MLP_on_selected_features(df, labels)

def RandomForest_on_selected_features(df, labels, features=None, best_alpha=0.001):
    if features:
        df = df.loc[:,features]
    return tune_RandomForest(df, labels, best_alpha=best_alpha)

def RandomForest_on_lasso_features(df, labels, best_alpha=0.001):
    df = get_lasso_reduced_data(df, labels)
    return RandomForest_on_selected_features(df, labels, best_alpha=best_alpha)

def RandomForest_on_pca_features(df, labels, n_components=None, pca_explain=None, best_alpha=0.001):
    df = get_pca_reduced_data(df, labels, n_components=n_components, pca_explain=pca_explain)
    return RandomForest_on_selected_features(df, labels, best_alpha=best_alpha)

def DecisionTree_on_selected_features(df, labels, features=None):
    if features:
        df = df.loc[:,features]
    return tune_DecisionTree(df, labels)

def DecisionTree_on_lasso_features(df, labels):
    df = get_lasso_reduced_data(df, labels)
    return DecisionTree_on_selected_features(df, labels)

def DecisionTree_on_pca_features(df, labels, n_components=None, pca_explain=None):
    df = get_pca_reduced_data(df, labels, n_components=n_components, pca_explain=pca_explain)
    return DecisionTree_on_selected_features(df, labels)

def SVM_on_selected_features(df, labels, features=None):
    if features:
        df = df.loc[:,features]
    return tune_SVM(df, labels)

def SVM_on_lasso_features(df, labels):
    df = get_lasso_reduced_data(df, labels)
    return SVM_on_selected_features(df, labels)

def SVM_on_pca_features(df, labels, n_components=None, pca_explain=None):
    df = get_pca_reduced_data(df, labels, n_components=n_components, pca_explain=pca_explain)
    return SVM_on_selected_features(df, labels)

def KNN_on_selected_features(df, labels, features=None):
    if features:
        df = df.loc[:,features]
    return tune_KNN(df, labels)

def KNN_on_lasso_features(df, labels):
    df = get_lasso_reduced_data(df, labels)
    return KNN_on_selected_features(df, labels)

def KNN_on_pca_features(df, labels, n_components=None, pca_explain=None):
    df = get_pca_reduced_data(df, labels, n_components=n_components, pca_explain=pca_explain)
    return KNN_on_selected_features(df, labels)

def get_lasso_reduced_data(df, labels):
    if not LASSO_DATA:
        lasso_data = ReducedDimensionsData(df, labels, red_method='lasso')
        lasso_data.reduce()
        lasso_df = lasso_data.data
        return lasso_df
    return LASSO_DATA

def get_pca_reduced_data(df, labels, n_components=None, pca_explain=None):
    if n_components:
        pca_reduced = ReducedDimensionsData(df, labels, n_dims=n_components)
    elif pca_explain:
        pca_reduced = ReducedDimensionsData(df, labels, pca_explain=pca_explain)
    else:
        pca_reduced = ReducedDimensionsData(df, labels)
    pca_reduced.reduce()
    pca_df = pca_reduced.get_data_df()
    return pca_df

def make_global_lasso_data(df, labels):
    global LASSO_DATA
    LASSO_DATA = ReducedDimensionsData(df, labels, red_method='lasso').reduce().get_data_df()

def classification_report_to_DataFrame(report, labels):
    df = pd.DataFrame(report).transpose()
    indices =labels.tolist()
    df2 = df.iloc[-2,:3]
    summary = df2.values
    df_sum = pd.DataFrame(summary).transpose()
    df_sum.columns=['precision','recall','f1-score']
    df_sum = df_sum.round(3)
    df = df[df.index.isin(indices)]               
    df = df.iloc[:,:3]
    return df


class ReducedDimensionsData():
    def __init__(self, df, labels, red_method='PCA', n_dims=None, pca_explain=0.9, lasso_max_iter=100) -> None:
        self.whole_data = df.sort_index()
        self.original_features = self.whole_data.columns
        self.original_index = self.whole_data.index
        self.labels = labels.sort_index()
        self.unique_labels = self.labels.unique()
        self.binary_labels = self.get_binary_labels()
        self.numeric_labels = self.get_numeric_labels()
        self.data = self.whole_data # Data used as input for the ML methods
        self.method = red_method
        self.pca_explain = pca_explain
        self.n_dims = n_dims
        self.description = ''
        self.whole_data_labelled = self.form_labelled_dataset(self.whole_data, self.labels)
        self.lasso_max_iter = lasso_max_iter

    def form_labelled_dataset(self, df, labels):
        return pd.concat([df, labels], axis = 1)

    def get_binary_labels(self):
        binary_labels, names = binarize_categorical(self.labels)
        return pd.DataFrame(binary_labels, columns=names)
    
    def get_numeric_labels(self):
        num_labels_dict = {k:v for (k,v) in zip(range(len(self.unique_labels)), self.unique_labels)}
        labels_num_dict = {v:k for (k,v) in num_labels_dict.items()}
        return self.labels.apply(lambda x: labels_num_dict[x])

    def set_description(self, description):
        self.description = description

    def reduce(self, standard=True, normalized=False, plot=False, test=False, verbose=False):
        if standard:
            self.standardize()
            self.data = self.scaled
        elif normalized:
            self.normalize()
            self.data = self.normalized
            
        if self.method=='PCA':
            if self.n_dims:
                pca = PCA(n_components=self.n_dims)
                pca.fit(self.data)
            else:
                current_explain = 0
                dims = 1
                while current_explain < self.pca_explain:
                    pca = PCA(n_components=dims)
                    pca.fit(self.data)
                    self.expl_vars = pca.explained_variance_ratio_
                    current_explain = sum(self.expl_vars)
                    dims += 1
                self.total_expl_var = current_explain
                self.pca_ndims = len(self.expl_vars)
                print('%.2f percent of data explained by %i components' % (self.total_expl_var*100, self.pca_ndims))
            self.data = pca.transform(self.whole_data)
            self.model = pca
        elif self.method=='lasso':
            count = 0
            while count < self.lasso_max_iter:
                pipeline = Pipeline([
                        ('scaler',StandardScaler()),
                        ('model',Lasso(max_iter=3000))
                ])
                search = GridSearchCV(pipeline,
                                    {'model__alpha':np.logspace(-4, 7, num=12)},
                                    cv = 10, scoring="neg_mean_squared_error")
                X_train, X_test, y_train, y_test = train_test_split(self.data, self.binary_labels, test_size=0.3)
                search.fit(X_train,y_train)
                best_alpha = search.best_params_['model__alpha']
                best_lasso = search.best_estimator_
                coefficients = search.best_estimator_.named_steps['model'].coef_
                importance = np.abs(coefficients)
                importance_sum = importance.sum(axis=0)
                relevant_features = np.array(self.data.columns)[importance_sum > 0]
                if verbose:
                    print('Best lasso alpha: ', best_alpha)
                    print('Lasso found %i relevant features:' % len(relevant_features))
                    print(relevant_features)
                    print('Lasso score: ', best_lasso.score(X_test, y_test))
                if (len(relevant_features) < 0.5 * len(self.data.columns)
                    and len(relevant_features) >= 1):
                    self.data = self.data.loc[:,relevant_features]
                    print('Lasso found %i relevant features:' % len(relevant_features))
                    print(relevant_features)
                    break
                count += 1
                self.model = best_lasso
        elif self.method == 'isomap':
            components = 2 if not self.n_dims else self.n_dims
            embedding = Isomap(n_components=components)
            self.data = embedding.fit_transform(self.whole_data)
            self.model = embedding

    def standardize(self, feats=None):
        if not feats:
            self.scaled = StandardScaler().fit(self.data).transform(self.data)
        else:
            self.scaled = StandardScaler().fit(self.data.loc[:,feats]).transform(self.data.loc[:,feats])
        self.scaled = pd.DataFrame(self.scaled, columns=self.original_features, index=self.original_index)

    def normalize(self, feats=None):
        if not feats:
            self.normalized = normalize(self.data, norm='l2')
        else:
            self.normalized = normalize(self.data.loc[:,feats], norm='l2')
        self.normalized = pd.DataFrame(self.normalized, columns=self.original_features, index=self.original_index)
    
    def get_data_df(self):
        if not type(self.data) == pd.DataFrame:
            return pd.DataFrame(self.data, index=self.original_index)
        else:
            return self.data


if __name__ == '__main__':
    #### Lasso selection
    lasso_data = ReducedDimensionsData(df_feats, data_labels, red_method='lasso')
    lasso_data.binary_labels
    lasso_data.reduce()
    lasso_df = lasso_data.data
    kmeans_lasso = apply_KMeans(lasso_df, 9)
    kmeans_lasso_assigned = kmeans_assignments(kmeans_lasso, lasso_df, data_labels)
    # PCA with 2 dims for the sake of plotting #####
    pca_2d = ReducedDimensionsData(df_feats, data_labels, red_method='PCA', n_dims=2)
    pca_2d.reduce()
    reduced_data = pca_2d.data
    ################################################
    plot_clusters_2d(kmeans_lasso_assigned, reduced_data, labels=None)

    #### PCA reduction
    pca_data = ReducedDimensionsData(df_feats, data_labels, pca_explain=0.75)
    pca_data.reduce()
    pca_df = pca_data.get_data_df()
    kmeans_pca = apply_KMeans(pca_df, 9)
    kmeans_pca_assigned = kmeans_assignments(kmeans_pca, pca_df, data_labels)
    pca_2d = ReducedDimensionsData(df_feats, data_labels, red_method='PCA', n_dims=2)
    pca_2d.reduce()
    reduced_data = pca_2d.data
    plot_clusters_2d(kmeans_pca_assigned, reduced_data, labels=None)