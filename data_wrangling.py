# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:47:15 2022

@author: remns
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.linear_model import Lasso


from plot_funcs import *
from funcs import *
from apply_ml import *
import specific_funcs

import pickle


PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
RESULTS_PATH = os.path.join(PROJECT_PATH, 'results')


# Indicate the data type of each numerical column (they are otherwise read as strings)
type_float = ['Rating', 'Displacement (ccm)', 'Power (hp)', 'Torque (Nm)',
              'Bore (mm)', 'Stroke (mm)', 'Fuel capacity (lts)',
              'Dry weight (kg)', 'Wheelbase (mm)', 'Seat height (mm)']
type_int = ['Year']

datatypes = {k:np.float64 for k in type_float}
for k in type_int:
    datatypes[k] = np.int32
    
# Load data
df = pd.read_csv(DATA_PATH+r'\all_bikez_curated.csv')
# Shorten category name for ease of plotting
df = change_categories_names(df, {'Prototype / concept model':'Prototype','Custom / cruiser':'Custom',\
                   'Cross / motocross':'Cross', 'Enduro / offroad':'Offroad'})
# Ensure numerical data type columns
df = df.astype(datatypes)

# List columns as numerical or categorical
features_num = ['Year', 'Displacement (ccm)', 'Power (hp)', 'Torque (Nm)',
              'Bore (mm)', 'Stroke (mm)', 'Fuel capacity (lts)',
              'Dry weight (kg)', 'Wheelbase (mm)', 'Seat height (mm)']
features_cat = [feat for feat in df.columns if feat not in features_num]

# Drop rows with unspecified category
unsp_ind = df['Category'] == 'Unspecified category'
df = df[~unsp_ind]
# Drop rows of underepresented categories
count_threshold = 200
counted = df.groupby('Category').count()
cat_to_drop = counted[counted.max(axis=1) < count_threshold].index.tolist()
index_to_drop = []
for cat in cat_to_drop:
    index_to_drop += df[df['Category'] == cat].index.tolist()
df = df.drop(index=index_to_drop)

# Clean useless columns
irrelevant_columns = ['Brand', 'Model', 'Rating', 'Color options']
dfc = df.drop(columns=irrelevant_columns)
features_num = [feat for feat in features_num if feat not in irrelevant_columns]
features_cat = [feat for feat in features_cat if feat not in irrelevant_columns]

# Create a reduced dat set that is more balanced
lowest_count = dfc.groupby('Category').count().max(axis=1).min()
n_random_rows = lowest_count * 2
row_blocks = []
for cat in dfc.groupby('Category').count().index:
    row_blocks.append(random_rows(dfc[dfc['Category']==cat], n_random_rows))
dfc_red = pd.concat(row_blocks)


# Select rows that contain all information (for both dfc and dfc_red)
conditions = [~dfc[col].isnull() for col in dfc.columns]
cond = conditions[0]
for c in conditions[1:]:
    cond = cond & c 
selected_rows = cond[cond==True]
df_sel = dfc.loc[selected_rows.index,:]
# Create a reduced data set from the complete rows that is more balanced
## a. Drop rows of underrepresented categories
count_threshold = 150
counted = df_sel.groupby('Category').count()
cat_to_drop = counted[counted.max(axis=1) < count_threshold].index.tolist()
index_to_drop = []
for cat in cat_to_drop:
    index_to_drop += df_sel[df_sel['Category'] == cat].index.tolist()
df_sel_filtered = df_sel.drop(index=index_to_drop)
## b. Random balanced selection
"""
with open('indices.p', 'rb') as fp:
    permanent_indices = pickle.load(fp)
dfc_red_sel = df_sel_filtered.loc[permanent_indices,:]
dfc_red_sel.sort_index(inplace=True)
"""
lowest_count = df_sel_filtered.groupby('Category').count().max(axis=1).min()
n_random_rows = lowest_count * 2
row_blocks = []
for cat in df_sel_filtered.groupby('Category').count().index:
    row_blocks.append(random_rows(df_sel_filtered[df_sel_filtered['Category']==cat], n_random_rows))
dfc_red_sel = pd.concat(row_blocks)
dfc_red_sel.sort_index(inplace=True)


categories = df_sel['Category'].unique()
engines = df_sel['Engine cylinder'].unique()

data_labels = dfc_red_sel['Category']
unique_labels = data_labels.unique()

num_labels_dict = {k:v for (k,v) in zip(range(len(unique_labels)), unique_labels)}
labels_num_dict = {v:k for (k,v) in num_labels_dict.items()}
numeric_labels = data_labels.apply(lambda x: labels_num_dict[x])
binary_labels, names = binarize_categorical(data_labels)

# Better leave this for the categorical analysis
# =============================================================================
# features_num.append('n_cylinders')
# df_sel['n_cylinders'] = df_sel['Engine cylinder'].apply(n_cylinders)
# dfc_red_sel['n_cylinders'] = dfc_red_sel['Engine cylinder'].apply(n_cylinders)
# 
# =============================================================================
print('Some EDA')
#fig, ax = bars_categories_count(df_sel, 'Category', title='Only complete rows')
#fig, ax = bars_categories_count(dfc_red_sel, 'Category', title='Only complete rows of reduced data')
#fig, ax = bars_categories_count(dfc_red, 'Category', title='Reduced data')
#fig, ax = bars_categories_count(df, 'Category', title='All data')

# EXPERIMENTS
df_cat = dfc_red_sel[features_cat]
df_num = dfc_red_sel[features_num]
df_num = rescale_years(df_num, col='Year', timerange=100)

num_norm = normalize(df_num, norm='l2')
pca = PCA(n_components=2)
pca.fit(num_norm)
print(pca.explained_variance_ratio_)
xnew = pca.transform(num_norm)
#plt.scatter(x=xnew[:,0], y=xnew[:,1])

num_scaled = StandardScaler().fit(df_num).transform(df_num)
pca2 = PCA(n_components=2)
pca2.fit(num_scaled)
print(pca2.explained_variance_ratio_)
xnew2 = pca2.transform(num_scaled)
#plt.scatter(x=xnew2[:,0], y=xnew2[:,1])


###############################################################################
##### Binarize the categorical columns #####
columns_to_binarize = ['Engine stroke', 'Gearbox', 'Fuel system', 'Fuel control',
                       'Cooling system', 'Transmission type', 'Front brakes',
                       'Rear brakes', 'Front tire', 'Rear tire', 'Front suspension',
                       'Rear suspension']
temporary_columns = []

######
fuelcontrol = df_cat['Fuel control'].unique()
######

# Engine cylinders to numeric
df_cat['n_cylinders'] = df_cat['Engine cylinder'].apply(n_cylinders)
temporary_columns.append('Engine cylinder')

# Binarization of the Engine stroke feature
df_cat = feature_to_binary(df_cat, 'Engine stroke')

# Work on the Fuel system feature
unique_categories = df_cat['Fuel system'].unique()
## Too many unique categories, find pattern and cluster them
sorted_words = find_main_words(df_cat['Fuel system'])
sorted_frequent_words = sorted_words[sorted_words>20]
#bar_plot_series(sorted_frequent_words)
discard_words = ['with','mm','mikuni','system','keihin','marelli','management',
                 'and','by','x','4', 'fuel']
sorted_few_words = sorted_frequent_words[[i for i in sorted_frequent_words.index if i not in discard_words]]
#bar_plot_series(sorted_few_words)

clusters, cluster_word_counter, clusters_sentences =\
    find_feature_categories(df_cat, 'Fuel system', sorted_few_words, 2)

df_cat, new_column = feature_transform_to_clusters(df_cat, 'Fuel system',
                                                   clusters_sentences)
temporary_columns.append(new_column)
df_cat = feature_to_binary(df_cat, 'Fuel system_reduced')

# Work on the Fuel control feature
unique_categories = df_cat['Fuel control'].unique()
fuelcontrol2 = df_cat['Fuel control'].unique()
unique_categories_count = count_unique_categories(df_cat, 'Fuel control')
## Unique categories are few enough such that we can binarize all of them
df_cat = feature_to_binary(df_cat, 'Fuel control')

# Work on the Cooling system feature
unique_categories = df_cat['Cooling system'].unique()
unique_categories_count = count_unique_categories(df_cat, 'Cooling system')
## Unique categories are few enough such that we can binarize all of them
df_cat = feature_to_binary(df_cat, 'Cooling system')

# Work on the Transmission type feature
unique_categories = df_cat['Transmission type'].unique()
unique_categories_count = count_unique_categories(df_cat, 'Transmission type')
## Unique categories are few enough such that we can binarize all of them
df_cat = feature_to_binary(df_cat, 'Transmission type')

# Work on the Front brake feature
unique_categories = df_cat['Front brakes'].unique()
unique_categories_count = count_unique_categories(df_cat, 'Front brakes')

df_cat['Front brakes'] = df_cat.loc[:,'Front brakes'].\
                            apply(lambda x: specific_funcs.unify_pistons(x))
df_cat = subfeature_from_feature(df_cat, 'Front brakes',
                                 'n_disks_front', {'drum':0,'single':1,'double':2})
df_cat = subfeature_from_feature(df_cat, 'Front brakes', 'abs',
                                 {'abs':1})
df_cat['abs'] = df_cat['abs'].fillna(value=0)
df_cat = subfeature_from_feature(df_cat, 'n_disks_front', 'has_front_disk',
                                 {'0':0,'1':1,'2':1})
df_cat = subfeature_from_feature(df_cat, 'Front brakes', 'n_pistons_front',
                                 {'1piston':1,'2piston':2,'3piston':3,'4piston':4,
                                  '6piston':6})
df_cat['n_pistons_front'] = df_cat['n_pistons_front'].fillna(value=1)

# Work on the Rear brake feature
unique_categories = df_cat['Rear brakes'].unique()
unique_categories_count = count_unique_categories(df_cat, 'Rear brakes')

df_cat['Rear brakes'] = df_cat.loc[:,'Rear brakes'].\
                            apply(lambda x: specific_funcs.unify_pistons(x))
df_cat = subfeature_from_feature(df_cat, 'Rear brakes',
                                 'n_disks_rear', {'drum':0,'single':1,'double':2})
df_cat = subfeature_from_feature(df_cat, 'Rear brakes', 'absrear',
                                 {'abs':1})
df_cat['absrear'] = df_cat['absrear'].fillna(value=0)
# Unify abs in one column #
df_cat['abs'] = df_cat[['abs', 'absrear']].max(axis=1)
df_cat.drop('absrear', axis=1, inplace=True)

df_cat = subfeature_from_feature(df_cat, 'n_disks_rear', 'has_rear_disk',
                                 {'0':0,'1':1,'2':1})
df_cat = subfeature_from_feature(df_cat, 'Rear brakes', 'n_pistons_rear',
                                 {'1piston':1,'2piston':2,'3piston':3,'4piston':4,
                                  '6piston':6})
df_cat['n_pistons_rear'] = df_cat['n_pistons_rear'].fillna(value=1)

# Work on Front tire and Rear tire features
# Tyre labelling code: 
    #width[mm]/aspect_ratio[%]-Speed[letter]Construction[letter]diameter[inch]
    # Construction can be either R, B or none (in which case it is B)
    # Speed: if Vxxx where xxx is a number, the number is the speed
    # Speed: Z is anything above 240 km/h, with no limit

df_cat = specific_funcs.tyres_columns(df_cat, 'Rear tire')
df_cat = specific_funcs.tyres_columns(df_cat, 'Front tire')
# Fill the Nan values based on bike class
# Width
df_cat = specific_funcs.fill_tyre_width(df_cat, 'Rear tire_width')
df_cat = specific_funcs.fill_tyre_width(df_cat, 'Front tire_width')
# Height
df_cat = specific_funcs.fill_tyre_height(df_cat, 'Rear tire_height')
df_cat = specific_funcs.fill_tyre_height(df_cat, 'Front tire_height')
# Speed
col_name = ' tire_speed'
df_cat.loc[:, 'Rear'+col_name] = df_cat.loc[:, 'Rear'+col_name].fillna(value='A')
df_cat.loc[:, 'Front'+col_name] = df_cat.loc[:, 'Front'+col_name].fillna(value='A')

# Construction
col_name = ' tire_construction'
df_cat.loc[:, 'Rear'+col_name] = df_cat.loc[:, 'Rear'+col_name].fillna(value='B')
df_cat.loc[:, 'Front'+col_name] = df_cat.loc[:, 'Front'+col_name].fillna(value='B')

# Diameter
df_cat = specific_funcs.fill_tyre_diameter(df_cat, 'Rear tire_diameter')
df_cat = specific_funcs.fill_tyre_diameter(df_cat, 'Front tire_diameter')

# Speed to numeric
speed_dict = specific_funcs.tyre_speed_and_construction(keys='lower')
avg = sum(speed_dict.values()) / len(speed_dict)
df_cat = subfeature_from_feature(df_cat, 'Rear tire_speed', 'reartyre_speed',
                                 speed_dict)
df_cat.loc[:, 'reartyre_speed'] = df_cat.loc[:, 'reartyre_speed'].fillna(value=avg)
temporary_columns.append('Rear tire_speed')

df_cat = subfeature_from_feature(df_cat, 'Front tire_speed', 'fronttyre_speed',
                                 speed_dict)
df_cat.loc[:, 'fronttyre_speed'] = df_cat.loc[:, 'fronttyre_speed'].fillna(value=avg)
temporary_columns.append('Front tire_speed')

# Construction to binary
df_cat = feature_to_binary(df_cat, 'Rear tire_construction')
df_cat = feature_to_binary(df_cat, 'Front tire_construction')
temporary_columns.append('Rear tire_construction')
temporary_columns.append('Front tire_construction')

# Format to binary
df_cat['Rear tire_label_format'] = df_cat['Rear tire_label_format'].fillna(value='I')
df_cat = feature_to_binary(df_cat, 'Rear tire_label_format')
df_cat['Front tire_label_format'] = df_cat['Front tire_label_format'].fillna(value='I')
df_cat = feature_to_binary(df_cat, 'Front tire_label_format')
temporary_columns.append('Rear tire_label_format')
temporary_columns.append('Front tire_label_format')

# Drop temporary columns and columns_to_binarize
df_cat = df_cat.drop(columns=temporary_columns+columns_to_binarize)


### Explore groups of columns that represent the same component
# Engine
control_cols = [c for c in df_cat.columns if 'Fuel control' in c]
fuel_system = [c for c in df_cat.columns if 'Fuel system' in c]
columns_engine = ['n_cylinders', 'Engine stroke_ four-stroke',
       'Engine stroke_ two-stroke', 'Engine stroke_Diesel', 'Cooling system_Air',
       'Cooling system_Liquid', 'Cooling system_Oil & air'] + control_cols +fuel_system
df_engine = df_cat.loc[:,columns_engine]
df_toscale = np.array(df_engine.loc[:,'n_cylinders']).reshape(-1,1)
engine_scaled = StandardScaler().fit(df_toscale).transform(df_toscale)
df_engine['n_cylinders'] = engine_scaled
pca = PCA(n_components=2)
pca.fit(df_engine)
print(pca.explained_variance_ratio_)
xnew = pca.transform(df_engine)
#plt.scatter(x=xnew[:,0], y=xnew[:,1])
pca = PCA(n_components=3)
pca.fit(df_engine)
print(pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))

### Let's reduce the Fuel Control features
# Visualize that some are clarly underrrepresented
#bar_plot_binary_features(df_cat, control_cols)
# Collect those underrepresented and put them all under Fuel control_Other
control_df = df_cat.loc[:, control_cols]
control_cols_sparse = control_df.loc[:,control_df.sum(axis=0)[control_df.sum(axis=0)<200].index.tolist()]
df_cat['Fuel control_Other'] = control_cols_sparse.sum(axis=1)
df_cat.drop(columns=control_cols, inplace=True)


###### FILL REMAINING NaN WITH SIMPLE METHOD

for col in df_cat.columns:
    if df_cat[col].isna().sum()>0:
        df_cat[col] = df_cat[col].fillna(method='ffill')

###################################################
#STANDARDIZE the non-binary features of df_cat
df_cat_feat = df_cat.loc[:,[c for c in df_cat.columns if c!='Category']]
cols_tostandard = df_cat_feat.max(axis=0)[df_cat_feat.max(axis=0)>1].index.tolist()
df_tostandard = df_cat.loc[:,cols_tostandard]
df_catstandard = StandardScaler().fit(df_tostandard).transform(df_tostandard)
df_cat[cols_tostandard] = df_catstandard
df_cat_feat = df_cat.loc[:,[c for c in df_cat.columns if c!='Category']]

pca = PCA(n_components=2)
pca.fit(df_cat_feat)
print(pca.explained_variance_ratio_)
xnew = pca.transform(df_cat_feat)
#plt.scatter(x=xnew[:,0], y=xnew[:,1])

# STANDARDIZE numeric features
df_num_feat = df_num.copy()
df_num_feat.loc[:,:] = StandardScaler().fit(df_num).transform(df_num)

df_cat_feat.sort_index(inplace=True)
df_num_feat.sort_index(inplace=True)

######## 2 VARIATIONS:
# 1- Categorical features are only standardized if they are not binary
df_feats = pd.concat([df_num_feat, df_cat_feat], axis = 1)

#2- All Categorical features are standardized: BETTER PERFORMANCE!!!
#df_cat_feat = df_cat.loc[:,[c for c in df_cat.columns if c!='Category']]
#df_cat_std = StandardScaler().fit(df_cat_feat).transform(df_cat_feat)
#df_cat_feat.loc[:,:] = df_cat_std
#df_cat_feat.sort_index(inplace=True)
#df_feats = pd.concat([df_num_feat, df_cat_feat], axis = 1)


id_to_index = {k:v for (k,v) in zip(df_feats.index, range(len(df_feats)))}
index_to_id = {v:k for (k,v) in id_to_index.items()}
true_groups = {}
for i,label in enumerate(unique_labels):
    true_groups[i] = {'samples':data_labels.index[data_labels==label].tolist()}
    true_groups[i]['samples_from_zero'] = [id_to_index[num] for num in true_groups[i]['samples']]
    true_groups[i]['category'] = label
true_labels = [v['category'] for v in true_groups.values()]

pca = PCA(n_components=2)
pca.fit(df_feats)
print(pca.explained_variance_ratio_)
x_feats = pca.transform(df_feats)

#plot_clusters_2d(true_groups, x_feats, true_labels)
#iso = ReducedDimensionsData(df_feats, data_labels, 'isomap')
#iso.reduce()
#iso_df = iso.get_data_df().to_numpy()
#plot_clusters_2d(true_groups, iso_df, labels=unique_labels, xmax=2000)


if False:
    kmeans = apply_KMeans(df_feats, n_clusters=9)
    kmeans_assigned = kmeans_assignments(kmeans, df_feats, data_labels)
    #plot_kmeans_2d(kmeans_assigned, xnew)


trainX, testX, trainY, testY = train_test_split(df_feats, data_labels, test_size = 0.2)
trainXarr, testXarr, trainYarr, testYarr = split_data_to_arrays(trainX, testX, trainY, testY)
"""
results = {}
count = 0

### DECISION TREE ##
tree_clf, tree_acc, report, conf_mat = tune_DecisionTree(df_feats, data_labels)
best_alpha = tree_clf.ccp_alpha
results['tree'] = {'cv_acc':tree_acc, 'report':report, 'confusion':conf_mat, 'params':None}

print('Done ', count)
count += 1

tree_lasso_clf, tree_lasso_acc, report, conf_mat = DecisionTree_on_lasso_features(df_feats, data_labels)
best_alpha_lasso = tree_lasso_clf.ccp_alpha
results['tree_lasso'] = {'cv_acc':tree_lasso_acc, 'report':report, 'confusion':conf_mat, 'params':None}

print('Done ', count)
count += 1

tree_pca_clf, tree_pca_acc, report, conf_mat = DecisionTree_on_pca_features(df_feats, data_labels)
best_alpha_pca = tree_pca_clf.ccp_alpha
results['tree_pca'] = {'cv_acc':tree_pca_acc, 'report':report, 'confusion':conf_mat, 'params':None} 

print('Done ', count)
count += 1

### MLP ###
#mlp_clf, mlp_acc_cv, report, conf_mat, params = tune_MLP(df_feats, data_labels, test_size=0.2)
#results['mlp'] = {'cv_acc':mlp_acc_cv, 'report':report, 'confusion':conf_mat, 'params':params}

mlp_clf = MLPClassifier(activation='tanh', alpha=0.05, early_stopping=False,
                        hidden_layer_sizes=(90,10), learning_rate='adaptive',
                        solver='adam', max_iter=3000)
mlp_clf.fit(trainXarr, trainYarr)
report_mlp = classification_report(testYarr, mlp_clf.predict(testXarr))
conf_mat = confusion_matrix(testYarr, mlp_clf.predict(testXarr))
results['mlp'] = {'cv_acc':None, 'report':report, 'confusion':conf_mat, 'params':None}

print('Done ', count)
count += 1

mlp_lasso_clf, mlp_lasso_acc, report, conf_mat, params = MLP_on_lasso_features(df_feats, data_labels)
results['mlp_lasso'] = {'cv_acc':mlp_lasso_acc, 'report':report, 'confusion':conf_mat, 'params':params}

print('Done ', count)
count += 1

mlp_pca_clf, mlp_pca_acc, report, conf_mat, params = MLP_on_pca_features(df_feats, data_labels)
results['mlp_pca'] = {'cv_acc':mlp_pca_acc, 'report':report, 'confusion':conf_mat, 'params':params}

print('Done ', count)
count += 1

### RANDOM FOREST ###
forest_clf, forest_acc, report, conf_mat = tune_RandomForest(df_feats, data_labels, test_size=0.2, best_alpha=best_alpha)  
results['rf'] = {'cv_acc':forest_acc, 'report':report, 'confusion':conf_mat, 'params':None}

print('Done ', count)
count += 1

forest_lasso_clf, forest_lasso_acc_, report, conf_mat = RandomForest_on_lasso_features(df_feats, data_labels)
results['rf_lasso'] = {'cv_acc':forest_lasso_acc_, 'report':report, 'confusion':conf_mat, 'params':None} 

print('Done ', count)
count += 1

forest_pca_clf, forest_pca_acc, report, conf_mat = RandomForest_on_pca_features(df_feats, data_labels)
results['rf_pca'] = {'cv_acc':forest_pca_acc, 'report':report, 'confusion':conf_mat, 'params':None} 

print('Done ', count)
count += 1

### NAIVE BAYES ###
nb_clf, nb_acc_, report, conf_mat = tune_NaiveBayes(df_feats, numeric_labels, test_size=0.2)
results['nb'] = {'cv_acc':nb_acc_, 'report':report, 'confusion':conf_mat, 'params':None} 

print('Done ', count)
count += 1

nb_lasso_clf, nb_lasso_acc, report, conf_mat = tune_NaiveBayes(df_feats, numeric_labels, test_size=0.2)
results['nb'] = {'cv_acc':nb_acc_, 'report':report, 'confusion':conf_mat, 'params':None} 

print('Done ', count)
count += 1

### SVC ###
svc_clf, svc_acc, report, conf_mat, params = tune_SVM(df_feats, data_labels, test_size=0.25)
results['svc'] = {'cv_acc':svc_acc, 'report':report, 'confusion':conf_mat, 'params':params}

print('Done ', count)
count += 1

svc_lasso_clf, svc_lasso_acc, report, conf_mat, params = SVM_on_lasso_features(df_feats, data_labels)
results['svc_lasso'] = {'cv_acc':svc_lasso_acc, 'report':report, 'confusion':conf_mat, 'params':params}

print('Done ', count)
count += 1

svc_pca_clf, svc_pca_acc, report, conf_mat, params = SVM_on_pca_features(df_feats, data_labels)
results['svc_pca'] = {'cv_acc':svc_pca_acc, 'report':report, 'confusion':conf_mat, 'params':params}

print('Done ', count)
count += 1

### KNN ###
knn_clf, knn_acc, report, conf_mat, params = tune_KNN(df_feats, data_labels)
results['knn'] = {'cv_acc':knn_acc, 'report':report, 'confusion':conf_mat, 'params':params}

print('Done ', count)
count += 1

knn_lasso_clf, knn_lasso_acc, report, conf_mat, params = KNN_on_lasso_features(df_feats, data_labels)
results['knn_lasso'] = {'cv_acc':knn_lasso_acc, 'report':report, 'confusion':conf_mat, 'params':params}

print('Done ', count)
count += 1

knn_pca_clf, knn_pca_acc, report, conf_mat, params = KNN_on_pca_features(df_feats, data_labels)
results['knn_pca'] = {'cv_acc':knn_pca_acc, 'report':report, 'confusion':conf_mat, 'params':params}

print('Done ', count)
count += 1

###  LINEAR SVM
lsvc_clf, lsvc_acc, report, conf_mat, params = tune_LSVM(df_feats, data_labels, test_size=0.2)
"""

"""
iso = ReducedDimensionsData(df_feats, data_labels, 'isomap', n_dims=3)
iso.reduce()
iso_df = iso.get_data_df().to_numpy()
fig, ax = plot_clusters_3d(true_groups, pca3d_data, labels=unique_labels, title='PCA 3 components', colormap='tab10', xmax=35)

pca3d = ReducedDimensionsData(df_feats, data_labels, 'PCA', n_dims=3)
pca3d.reduce()
pca3d_data = pca3d.get_data_df().to_numpy()
fig, ax = plot_clusters_3d(true_groups, pca3d_data, labels=unique_labels, title='PCA 3 components', colormap='Set1', xmax=35)
"""
    
    
    