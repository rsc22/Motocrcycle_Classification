
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score

from scipy.special import softmax


def n_cylinders(word):
    if word in ['Single cylinder']:
        return 1
    elif word in ['V2', 'Twin', 'Two cylinder boxer']:
        return 2
    elif word in ['In-line three', 'Diesel']:
        return 3
    elif word in ['V4', 'In-line four', 'Four cylinder boxer']:
        return 4
    elif word in ['In-line six', 'Six cylinder boxer']:
        return 6
    elif word in ['Electric']:
        return 0
    
def ID_from_array_index(index, df_origin):
    return df_origin.index[index]

def find_main_words(feature):
    sentences = feature.unique().tolist()
    sentences_split = [get_words(s) for s in sentences]
    words_count = count_words(sentences_split)
    sorted_words = pd.Series(words_count)
    sorted_words.sort_values(ascending=False, inplace=True)
    return sorted_words

def change_categories_names(df, names_dict):
    df['Category'] = df.loc[:,'Category'].apply(lambda x: names_dict[x] if x in names_dict.keys() else x)
    return df

def random_rows(df, n_indices):
    indices = df.index.tolist()
    if len(indices) <= n_indices:
        return df.copy()
    rnd_indices = np.random.choice(indices, size=n_indices, replace=False)
    rnd_rows = df.loc[rnd_indices,:]
    return rnd_rows

def form_labelled_dataset(df, labels):
        return pd.concat([df, labels], axis = 1)

def rescale_years(df, col='Year', timerange=100):
    min_year = df.loc[:,col].min()
    subtract = min_year - timerange
    df.loc[:,col] = df.loc[:,col] - subtract
    return df

def binarize_dual(feature):
    """Receives a feature column with binary labels (only two classes)
    Returns a single feature column of zeros and ones and the name of the class encoded with 1 (as a list)"""
    unique_vals = feature.unique()
    cat = unique_vals[0]
    bin_array = np.zeros((len(feature), ))
    bin_array[np.where(feature==cat)] = 1
    bin_array[np.where(feature!=cat)] = 0
    return bin_array, [cat]

def binarize_categorical(feature):
    """Receives a vector of a categorical feature.
    Returns the matrix of the binarized feature and the names of each category
    (which are the columns in the matrix). If the feature has just two classes,
    return one binary column"""
    unique_vals = feature.unique()
    if len(unique_vals) == 2:
        bin_array, unique_vals = binarize_dual(feature)
    else:
        bin_array = np.zeros((len(feature), len(unique_vals)))
        for j, cat in enumerate(unique_vals):
            bin_array[np.where(feature==cat),j] = 1
            bin_array[np.where(feature!=cat),j] = 0
    return bin_array, unique_vals

def feature_to_binary(df, col_name):
    """Create binary vectors corresponding to each possible category in the feature
    represented by the column passed.
    Return the received DataFrame, with the new columns which are the binary arrays"""
    feature = df.loc[:,col_name]
    bin_array, names = binarize_categorical(feature)
    names = [col_name+'_'+str(n) for n in names]
    if len(names) > 1:
        bin_array_t = bin_array.T
        for i, col_t in enumerate(bin_array_t):
            df.loc[:,names[i]] = col_t.T
    else:
        df.loc[:,names[0]] = bin_array.T
    return df

def count_unique_categories(df, col_name):
    counts = df.groupby(col_name).count()
    series_count = pd.Series(counts.iloc[:,0], index=counts.index)
    return series_count

def character_rules(c, special_accepted=[r'-', r'/', '.']):
    if c.isalnum():
        return True
    for special in special_accepted:
        if c == special:
            return True
    return False

def get_words(sentence):
    """Return a list of the words found in the sentence"""
    words = sentence.split()
    for c in ['.', ',', ';', '(', ')', '-']:
        for i,word in enumerate(words):
            word = word.replace(c, '')
            word = word.lower()
            words[i] = word
    return words

def count_words(sentences):
    """sentences must be a collection of lists of words (strings)"""
    words_count = {}
    for s in sentences:
        s = set(s)
        for word in s:
            if word in words_count.keys():
                words_count[word] += 1
            else:
                words_count[word] = 1
    return words_count

def find_words_pair(sentences, word1, word2):
    """Find the sentences in sentences where both words appear.
    Sentences is either a list of lists of words or a list of cleaned/formatted strings.
    Returns the list of indices of the found sentences in the original snetences list
    and the list of sentences found"""
    indices, examples = [], []
    for i, s in enumerate(sentences):
        s = s.lower()
        if word1 in s and word2 in s:
            indices.append(i)
            examples.append(s)
    return indices, examples

def count_words_pairs(sentences, relevant_words):
    """Returns an WHEIGHTED ADJACENCY MATRIX of words based on the count of
    observations where both words are present"""
    sentences = [s.lower() for s in sentences]
    if type(relevant_words) == pd.Series:
        words = relevant_words.index.tolist()
    words = [w.lower() for w in words]

    A = np.zeros((len(words), len(words)))
    for i, word1 in enumerate(words):
        try:
            j = i+1
            rest_words = words[j:]
            for word2 in rest_words:
                #print(str(i)+' '+str(j)+' --> '+word1+'+'+word2)
                ix, _ = find_words_pair(sentences, word1, word2)
                A[i,j] = len(ix)
                j += 1
        except IndexError:
            break
    A = np.triu(A) + np.tril(A.T, k=1)
    return A

def sentences_where_word(sentences, word):
    if type(sentences) == pd.Series:
        sentences = sentences.index.tolist()
    sentences = [s.lower() for s in sentences]
    found = [s for s in sentences if word in s]
    return found

def sentences_where_not_word(sentences, word):
    if type(sentences) == pd.Series:
        sentences = sentences.index.tolist()
    sentences = [s.lower() for s in sentences]
    found = [s for s in sentences if word not in s]
    return found

def categories_embedding(sentences, relevant_words):
    """Returns a matrix whose columns are embeddings of the sentences, 
    encoded by the presence or not of the words in the relevant_words bag"""
    sentences = [s.lower() for s in sentences]
    if type(relevant_words) == pd.Series:
        words = relevant_words.index.tolist()
    words = [w.lower() for w in words]

    embedding = np.zeros((len(sentences), len(words)))
    for i, s in enumerate(sentences):
        array = np.zeros((len(words),))
        for j, word in enumerate(words):
            if word in s:
                array[j] = 1
        embedding[i,:] = array
    return embedding

def find_feature_categories(df, col_name, relevant_words, n_cat=3, reduce_dim=True, n_dim=4):
    """Return groups (clusters) of sentences (which are unique categories) from the
    given feature (col_name in df).
    Choose whether to cluster the raw sentences or to reduce dimensionality first using PCA"""
    embedding = categories_embedding(df[col_name].unique().tolist(), relevant_words)
    if reduce_dim:
        pca = PCA(n_components=n_cat)
        pca.fit(embedding)
        resultpca = pca.transform(embedding)
    if reduce_dim:
        clusters = do_clustering(resultpca, n_cat)
    else:
        clusters = do_clustering(embedding, n_cat)
    clusters_sentences = get_sentences_by_cluster(clusters, df[col_name].unique().tolist())      
    cluster_word_counter = get_word_count_by_cluster(clusters_sentences, relevant_words)
    return clusters, cluster_word_counter, clusters_sentences

def do_clustering(data, n_clusters, method='KMeans'):
    if method == 'KMeans':
        clustering = KMeans(n_clusters=n_clusters, random_state=0)
    elif method=='SpectralClustering':
        clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=0)
    clusters = clustering.fit_predict(data)
    return clusters

def get_sentences_by_cluster(clusters, sentences):
    clusters_sentences = {k:[] for k in np.unique(clusters)}
    for i, s in enumerate(sentences):
        cluster = clusters[i]
        clusters_sentences[cluster].append(s)
    return clusters_sentences

def get_word_count_by_cluster(clusters_sentences, relevant_words):
    cluster_word_counter = {}
    for i, cluster in clusters_sentences.items():
        cluster_word_counter[i] = {k:0 for k in relevant_words.index}    
        for s in cluster:
            s = s.lower()
            for word in relevant_words.index:
                if word in s:
                    cluster_word_counter[i][word] += 1
                    
    for k, v in cluster_word_counter.items():
        cluster_word_counter[k] = pd.Series(v).sort_values(ascending=False)
    return cluster_word_counter

def feature_transform_to_clusters(df, col_name, clusters_sentences):
    """Add a column to the DataFrame recording the cluster to which each entry of the
    feature column belongs"""
    new_column = col_name+'_reduced'
    df[new_column] = df.loc[:,col_name].apply(lambda x: map_feature_to_cluster(x, clusters_sentences))
    return df, new_column

def map_feature_to_cluster(s, clusters_sentences):
    for k, v in clusters_sentences.items():
        if s in v:
            return k

def find_keyword_return_value(s, mapping):
    """Mapping keys must always be strings"""
    if type(s) != str:
        s = str(s)
    s = s.lower()
    for k in mapping.keys():
        if k in s:
            return mapping[k]
    return np.nan

def subfeature_from_feature(df, col_name, new_column, mapping):
    """Mapping keys must always be strings"""
    df[new_column] = df.loc[:,col_name].apply(lambda x: find_keyword_return_value(x, mapping))
    return df


def kmeans_assignments(kmeans, df, true_classes):
    """kmeans: KMeans object trained on df.
        df: DataFrame of the features
        true_classes: pandas Series with same index as df and true data categories as values"""
    """
    if type(id_dict) != dict:
        if type(id_dict) == list or type(id_dict) == tuple:
            standard_labels = [i for i in range(len(id_dict))]
            id_dict = {k:v for (k,v) in zip(standard_labels, id_dict)}
        else:
            raise TypeError('id_dict has incorrect type. Admitted types are'
                'dict, list, tuple')
    """
    cluster_pred = kmeans.predict(df)
    cluster_ids = np.unique(cluster_pred)
    centers = kmeans.cluster_centers_
    id_to_index = {k:v for (k,v) in zip(df.index, range(len(df)))}
    index_to_id = {v:k for (k,v) in id_to_index.items()}
    clusters = {}
    for id in cluster_ids:
        clusters[id] = {'samples':None, 'samples_from_zero':None, 'majority':None, 'majority_perc':None}
        clusters[id]['samples'] = df.index[np.where(cluster_pred == id)].to_series()
        clusters[id]['samples_from_zero'] = [id_to_index[num] for num in clusters[id]['samples']]

    # Find out what is the majority true label in each cluster
    for id, cluster in clusters.items():
        labels = true_classes[cluster['samples']]
        labels_count = labels.value_counts()
        clusters[id]['majority'] = labels_count

    # If a label is the majority in several clusters, assign it only to the
    # cluster where its percentual weight is bigger. Then assign the second
    # most frequent class as the label of the other cluster
    """
    classes = true_classes.unique()
    for cat in classes:
        class_count = 0
        percentages = {}
        for k, v in clusters.items():
            if v['majority'] == cat:
                class_count += 1
                percentages[k] = v['majority_perc']
        if class_count > 1:
    """
    return clusters

def kmeans_df(kmeans, data_df, data_labels):
    clusters = kmeans_assignments(kmeans, data_df, data_labels)
    counts = []
    for cluster in clusters.values():
        counts.append(cluster['majority'])
    df_kmeans = pd.concat(counts, axis=1)
    df_kmeans.columns = ['Cluster '+str(i) for i in range(0, len(df_kmeans.columns))]
    df_kmeans = df_kmeans.fillna(value=0)
    return df_kmeans




