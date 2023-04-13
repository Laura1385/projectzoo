import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
import scipy as sp
import warnings
warnings.filterwarnings('ignore')
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA           #lo usi o no?

def border_msg(msg):
    row = len(msg)
    h = ''.join(['+'] + ['-' *row] + ['+'])
    result= h + '\n''|'+msg+'|' '\n' + h
    print(result)

def analisys(filename):
    df = pd.read_csv(filename,sep=',')
    border_msg('Nome File:')
    print(filename, "\n")
    border_msg('Informazioni DataFrame:')
    df.info()
    print('\n - Numero di osservazioni (righe) x caratteristiche (colonne):', df.shape, '\n\n')
    print(border_msg('Anteprima del DataFrame:'),df.head(5), '\n')
    print(border_msg('Statistiche di base:'),' ',df.describe(), '\n')
    print(border_msg('Analisi variabili:'),'\n', df.nunique(), '\n\n')
    dfname= str(filename) 
    name= dfname[:-4]#take values before the point
    prefix= 'df_'
    f = print('Rinomino il DataFrame creato in:', prefix + name)

def confusion_matrix(df):
    #Import data
    data = df
    corr = data.corr() #osx need (numeric_only=True)
    #Print data
    #print(data.corr())
    #Create space of the graphic 
    fig, ax = plt.subplots(figsize=(7, 4))
    #Create the mask to NOT have graphic redundancy
    mask = np.triu(np.ones_like(corr, dtype=bool), k = 1)
    #Create correlation palette and heat map
    cmap = sns.color_palette('Spectral_r', as_cmap=True)
    dataplot = sns.heatmap(corr, mask=mask, center=0, annot=True, annot_kws={'size': 8}, fmt='.1f', cmap=cmap, linewidths=.35)
    plt.title(f'Correlation Matrix Animals', fontsize=15)
    plt.show()
    
def update_animal_name(df, old_name, new_name):
    index = df.index[df['animal_name (nome_animale)'].str.startswith(old_name)]
    if len(index) > 0: #cerca il nome
        old_value = df.at[index[0], 'animal_name (nome_animale)']
        new_full_name = old_value.replace(old_name, new_name)
        df.at[index[0], 'animal_name (nome_animale)'] = new_full_name
        print(f"Ho modificato '{old_value}' con '{new_full_name}' ")
    else:
        print(f"Nessun animale trovato che inizi con '{old_name}' ")

def modify_column(df, animal_column, column_to_modify, values_to_modify):
    for animal, value in values_to_modify.items():
        # values_to_modify is a dictionary with animal's name like keys to modify and their new values like value
        df.loc[df[animal_column] == animal, column_to_modify] = value
    print("Le modifiche indicate sono state apportate")
        
def t_sne(data, n_components=2, n_iter=500, n_iter_without_progress=150, n_jobs=2, random_state=0):
    tsne = TSNE(n_components=n_components, 
           n_iter=n_iter, 
           n_iter_without_progress=n_iter_without_progress,
           n_jobs=n_jobs, 
           random_state=random_state)
    data_reduced = tsne.fit_transform(data)
    return np.array(data_reduced)
    print(data_reduced)

def scatter_plot(X, c, title, cmap='Accent', marker='o', edgecolor=None, colorbar=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()
    im = ax.scatter(X[:, 0], X[:, 1], c=c, cmap=cmap, marker=marker, edgecolor=edgecolor)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        fig.colorbar(im, ax=ax)
    ax.set_title(title)
    return fig, ax
   
def km_clust(X, n_clusters, init, n_init=10, max_iter=300, tol=1e-4, random_state=0):
    kmeans = KMeans(n_clusters=n_clusters, 
                    init=init, n_init=n_init, 
                    max_iter=max_iter, tol=tol, 
                    random_state=random_state)
    y_km_clustering = kmeans.fit_predict(X)+1 #correct labels(now start from 1, not 0)
    
    #Stampa le prime 10 istanze clusterizzate e le etichette univoche dei cluster.
    #print(f"Primi 10 punti clusterizzati: {y_km_clustering[:10]}")
    print(f"Etichette univoche dei cluster: {np.unique(y_km_clustering)}")
    return kmeans, y_km_clustering

def plot_kmean(X, labels, centroids, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()
    ax.set_title(title)
    for label in np.unique(labels):
        ax.scatter(
            *X[labels == label].T,
            marker='o',
            s=60,
            color=plt.cm.tab10(label),
            alpha=0.8,
        )
    if len(centroids) > 0:
        ax.scatter(centroids[:,0], centroids[:,1],
            s=250, marker='*',
            c='yellow', edgecolor='red',
            label='Centroidi'
        )
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])  

def results_group(data, df, y, nome_algoritmo):
    #find animal's name inside each cluster
    results = []
    for index, a in enumerate(data):
        #results.append('{},{}'.format(df_new2.loc[index,'animal_name (nome_animale)'], y_km_clustering[index]))
        results.append(f"{df.loc[index, 'animal_name (nome_animale)']},{y[index]}")
    #print(results)

    #tuple lists with data split in name e cluster
    new_list = [tuple(x.split(",")) for x in results]

    #newdf with 2 coloums
    df_cl = pd.DataFrame(new_list, columns=['Animal', 'Pred Labels'])

    #animal in every cluster
    group_cl = df_cl.groupby(['Pred Labels'])
    print('Algoritmo', nome_algoritmo, '\n')
    for num, (nome, gruppo) in enumerate(group_cl):
        print('Pred Labels', num+1)
        animali = list(gruppo['Animal'])
        for x in animali:
            print(x)
        print('Totale:', len(gruppo),'\n')   
    return df_cl  
    
def plot_silhouette(X, kmeans, title='Silhouette Plot'):
    cluster_labels = np.unique(kmeans.labels_) #array of labels (ex kmplus_clustering)
    n_clusters =  len(cluster_labels)
    silhouette_vals = silhouette_samples(X, kmeans.labels_, metric='euclidean')#silhoutte for each sample (ex kmplus_clustering)
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[kmeans.labels_ == c] #get the s(i) for samples of cluster c
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = plt.cm.tab10(c)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
                 edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--", label='Average') 
    plt.yticks(yticks, cluster_labels + 1)
    plt.title(title)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.xlim((-0.25,1))
    plt.legend()
    plt.show()
    print('Media:%.3f' % silhouette_avg)  
    
def agglomerative_clust(X, linkage, n_clusters=7, affinity='euclidean'):
    ac = AgglomerativeClustering(n_clusters=n_clusters,
                                 affinity=affinity,
                                 linkage=linkage)
    y_ac_blob = ac.fit_predict(X)+1 #correct labels(now start from 1, not 0)
    print(f"Etichette univoche dei cluster: {np.unique(y_ac_blob)}")
    return ac, y_ac_blob

def dbscan_clust(X, eps= 0.5, min_samples= 5, metric='euclidean'):
    dbscan = DBSCAN(eps=eps, 
                    min_samples=min_samples, 
                    metric='euclidean')
    labels = dbscan.fit_predict(X)+2 #correct labels(now start from 1, not 0)
    print(f"Etichette univoche dei cluster: {np.unique(labels)}")
    return dbscan, labels