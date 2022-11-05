# %%


# %%
import numpy as np
import pandas as pd
import matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
sns.set_theme()
sns.set_palette("Greens_d")
sns.light_palette("seagreen", as_cmap=True)
sns.color_palette("Greens_d")

#import plotly.express as px
import tarfile
import json
import csv
import random
import time
import re

#import cv2

from datetime import datetime

# Import custom helper libraries
import os
#from os import listdir, path
#from os.path import isfile, join, splitext

import sys
#import data.helpers as data_helpers
#import visualization.helpers as viz_helpers

# from joblib import dump, load
import pickle

#from PIL import Image

# import tensorflow as tf
# from tensorflow import keras

# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences


# %%
# function
def Test_Imported_Functions():
    print("Functions have been properly imported !")

# %% [markdown]
# ## Load data

# %%
def load_nrows_data(DATA_URL, nrows, DATASET_COLUMNS):
    data_loaded = pd.read_csv(DATA_URL, nrows=nrows, names=DATASET_COLUMNS)
    return (data_loaded)

def load_all_data(DATA_URL, DATASET_COLUMNS):
    data_loaded = pd.read_csv(DATA_URL, names=DATASET_COLUMNS)
    return (data_loaded)

def load_formatted_data(DATA_URL, DATASET_ENCODING, DATASET_COLUMNS):
    data_loaded = pd.read_csv(DATA_URL, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
    return (data_loaded)


# %% [markdown]
# ## Clean data

# %%
def clean_data(data): 
    #Remove rows where important information are missing:
    data_cleaned = data.dropna(axis = 0, how='all')
    #Clean duplicates
    data_cleaned = data_cleaned.drop_duplicates()
    #Change content in lowercase
    data_cleaned = data_cleaned.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
    #Filter order_dataset
    return (data_cleaned)

# %% [markdown]
# ## Plot Dataframe

# %%


# %%
def colors_from_values_integer(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

# def colors_from_values(values, palette_name):
#     pal = sns.color_palette(palette_name, len(values))
#     rank = values.argsort().argsort()   # http://stackoverflow.com/a/6266510/1628638
#     palette=np.array(pal[::-1])[rank]
#     return (palette)

def colors_from_values_float(values: pd.Series, palette_name:str, ascending=True):
    # convert to indices
    values = values.sort_values(ascending=True).reset_index()
    indices = values.sort_values(by=values.columns[0]).index
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

# %%
#Function to plot Fill ratio in specified columns
def plot_fill_ratio(data: pd.DataFrame, colunms_selected: list):
    data_fill_ratio = pd.DataFrame(columns=['column_name', 'null_count', 'notnull_count'])
    data_fill_ratio.drop(data_fill_ratio.index, inplace=True)        
    for col in colunms_selected: 
        null_count = data[col].isna().sum()
        notnull_count = data[col].notna().sum()
        new_row = pd.DataFrame({'column_name':[col], 'null_count':[null_count], 'notnull_count':[notnull_count]})
        data_fill_ratio = pd.concat([data_fill_ratio, new_row], ignore_index = None, axis = 0)
    data_fill_ratio_study = pd.melt(data_fill_ratio.reset_index(), id_vars=['column_name'], value_vars=['null_count', 'notnull_count'])
    fig, ax = plt.subplots(figsize=(16,10))
    #ax = sns.barplot(data=data_fill_ratio_study, x='value', y='column_name', hue='variable')
    
    ax = sns.barplot(data=data_fill_ratio_study, x='value', y='column_name', hue='variable', palette="Greens_d")
    ax.set_title('Null and NotNull Count per columns in dataframe')
    plt.show()
    data_fill_ratio_study.drop(data_fill_ratio_study.index, inplace=True)
    return(data_fill_ratio)



# %%
#Function to plot occurence by value present in specified column
def plot_occurence_line(data: pd.DataFrame, colunm_name):
    fig = px.line(data[colunm_name].value_counts())
    fig.update_layout(
        title_text=f"Number of occurence by {colunm_name} .\nTOTAL = {len(data[colunm_name])}",
        width=900,
        height=600,
        #markers=True,
    )
    fig.show()

# %%
#Function to plot distribution of dates
def plot_peryearmonth(data: pd.DataFrame, date_column, plot_hue: bool, hue_column):
    data['date_yearmonth'] = pd.to_datetime(data[date_column]).dt.to_period('M')
    plt.figure(figsize=(15,10))
    if (plot_hue == True):
        ax1 = sns.countplot(x="date_yearmonth", data=data.sort_values('date_yearmonth'), hue=hue_column, palette="Greens_d")
    else:
        ax1 = sns.countplot(x="date_yearmonth", data=data.sort_values('date_yearmonth'), palette="Greens_d")
    ax1.set_title(f'Distribution of {date_column} per month')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.show()

#Function to plot distribution of dates
def plot_peryear(data: pd.DataFrame, date_column, plot_hue: bool, hue_column):
    data['date_year'] = pd.to_datetime(data[date_column]).dt.to_period('Y')
    plt.figure(figsize=(15,10))
    if (plot_hue == True):
        ax1 = sns.countplot(x="date_year", data=data.sort_values('date_year'), hue=hue_column, palette="Greens_d")
    else:
        ax1 = sns.countplot(x="date_year", data=data.sort_values('date_year'), palette="Greens_d")
    ax1.set_title(f'Distribution of {date_column} per year')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.show()

# %%
def pairplot_columns(data: pd.DataFrame, colunms_selected: list, plot_hue: bool, hue_column):
    #fig, ax = plt.subplots(figsize=(15,10))
    if (plot_hue == True):
        ax = sns.pairplot(data[colunms_selected], 
                             hue=hue_column, 
                             hue_order=sorted(data[hue_column].unique(),
                             reverse=True)
                            )
    else:
        ax = ax=sns.pairplot(data[colunms_selected]
                            )
    ax.fig.suptitle(f'Pairplot on selected columns')
    plt.title(f'Pairplot on selected columns {colunms_selected}')
    plt.show()

# %%
#Function to plot PIE Chart of n tops values in dataframe
def plot_ntops_pie(data: pd.DataFrame, colunm_name, ntops: int, plot_others: bool, plot_na: bool):
    podium_tops = pd.DataFrame(data[colunm_name].value_counts(dropna=True, sort=True).head(ntops))
    if (plot_others == True):
        remainings_counts = sum(data[colunm_name].value_counts(dropna=True)[ntops:])
        remainings_below = pd.DataFrame({colunm_name : [remainings_counts]}, index=['others'])
        podium_tops = pd.concat([podium_tops, remainings_below], ignore_index = None, axis = 0)
    if (plot_na == True):
        na_counts = data[colunm_name].isna().sum()
        remainings_na = pd.DataFrame({colunm_name : [na_counts]}, index=['NAN'])
        podium_tops = pd.concat([podium_tops, remainings_na], ignore_index = None, axis = 0)
    
    
    #Définir la taille du graphique
    plt.figure(figsize=(8,8))
    #Définir lae type du graphique, ici PIE CHart avec en Labels l'index du nom des libelle
    #l'autopct sert ici à afficher le % calculé avec 1 décimal derriere la virgule
    plt.pie(podium_tops[colunm_name], labels=podium_tops.index, autopct='%1.1f%%')
    #Afficher la légende en dessous du graphique au centre
    plt.legend(loc='upper left', bbox_to_anchor=(0.1, -0.01), fancybox=True, shadow=None, ncol=2)
    plt.title(f"{ntops} most presents values identified in column {colunm_name} .\nTOTAL unique = {len(data[colunm_name].unique())}")
    #Afficher le graphique
    plt.show()
    return(podium_tops)



# %%
def plot_ntops_bar(data: pd.DataFrame, colunm_name, ntops: int, plot_others: bool, plot_na: bool):
    podium_tops = pd.DataFrame(data[colunm_name].value_counts(dropna=True, sort=True).head(ntops))
    if (plot_others == True):
        remainings_counts = sum(data[colunm_name].value_counts(dropna=True)[ntops:])
        remainings_below = pd.DataFrame({colunm_name : [remainings_counts]}, index=['others'])
        podium_tops = pd.concat([podium_tops, remainings_below], ignore_index = None, axis = 0)
    if (plot_na == True):
        na_counts = data[colunm_name].isna().sum()
        remainings_na = pd.DataFrame({colunm_name : [na_counts]}, index=['NAN'])
        podium_tops = pd.concat([podium_tops, remainings_na], ignore_index = None, axis = 0)
    #podium_tops = podium_tops.reset_index(drop=True)
    #Définir la taille du graphique
    fig, ax = plt.subplots(figsize=(15,10))
    #Définir lae type du graphique, ici BARPLOT avec en Labels l'index du nom des libelle
    ax = sns.barplot(data=podium_tops, x=podium_tops.index, y=colunm_name, palette=colors_from_values_integer(podium_tops[colunm_name], "Greens_d"))
    plt.title(f"{ntops} most presents values identified in column {colunm_name} .\nTOTAL unique = {len(data[colunm_name].unique())}")
    #Afficher le graphique
    plt.show()
    return(podium_tops)


# %%
#Create function that study boxplot
def plot_boxplot(data: pd.DataFrame, x_axis, colunms_selected: list, plot_outliers: bool): 
    for col in colunms_selected:
        sns.set()
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.boxplot(x=x_axis, 
                    y=col, # column is chosen here
                    data=data,
                    #order=["a", "b"],
                    showfliers = plot_outliers,
                    showmeans=True,
                    )  
        sns.despine(offset=10, trim=True) 
        plt.show()
        

# %%
#Create function that study histogramme
def plot_histogramme(histo_data: pd.DataFrame, column_value, colunms_group): 
    fig, ax = plt.subplots(figsize=(15, 5))
    #Plot the distribution
    ax = sns.displot(data=histo_data, x=column_value, hue=colunms_group)
    ax.move_legend(ax1, "upper right", bbox_to_anchor=(.55, .45), title=f'histogramme of {column_value}')
    plt.title(f"Distribution of {column_value} values")
    #plt.legend(loc='upper right')
    plt.ylabel("Count")
    plt.xlabel(f"{column_value} ranges")
    plt.show()

# %% [markdown]
# ## Réduction de dimension
# ### ACP (A Vérifier)

# %%
#PCA functions:
#Functions below are used for ACP
def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(8,8))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='r')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

# %%
#Functions below are used for ACP
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(8,8))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
                #plt.scatter(centres_reduced[:, d1], centres_reduced[:, d2], alpha=alpha, marker='x', s=100, linewidths=2,color='k', zorder=10)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                    #plt.scatter(centres_reduced[:, d1], centres_reduced[:, d2], alpha=alpha, marker='x', s=100, linewidths=2,color='k', zorder=10)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) / 1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)


# %%
#Functions below are used for ACP            
def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)


# %%
def display_parallel_coordinates(df, num_clusters):
    '''Display a parallel coordinates plot for the clusters in df'''

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.cluster==i])
    
    # Create the plot
    fig = plt.figure(figsize=(12, 15))
    title = fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)
    fig.subplots_adjust(top=0.95, wspace=0)

    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters
    for i in range(num_clusters):    
        plt.subplot(num_clusters, 1, i+1)
        for j,c in enumerate(cluster_points): 
            if i!= j:
                pc = parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j],0.2)])
        pc = parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i],0.5)])

        # Stagger the axes
        ax=plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20)        


# %%
def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20) 

# %%
#Functions below are used for Data Clustering    
def plot_dendrogram(linked, names):
    plt.figure(figsize=(10,15))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        linked,
        labels = names,
        orientation = "left",
        show_leaf_counts=True
    )
    plt.show()


# %% [markdown]
# ### Matrice de Confusion

# %%
#Fonction pour le graphe de Matrice de Confusion
def matrix_pred_model(model, model_name, y_test, y_pred, figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.title(f"Matrice de confusion de {model_name}")
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap = 'Greens',fmt="d",cbar=False)
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe initiale")
    plt.show()

# %%
def plot_roc_auc_curve(model_name, fpr, tpr, figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.title(f"ROC Curve for {model_name}")
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=model_name)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    plt.show()

# %% [markdown]
# ### Determiner le seuil de décision (Classification binaire)

# %%
def plot_threshold_scores(data: pd.DataFrame, optimized_seuil):
    plt.figure(figsize=(10,5))
    plt.plot(data['seuil'], data['FBeta-Score'], color='coral', lw=2, label='FBeta-Score')
    plt.plot(data['seuil'], data['Precision_score'], color='cyan', lw=2, label='Precision_score')
    plt.plot(data['seuil'], data['Accuracy_score'], color='blue', lw=2, label='Accuracy_score')
    plt.plot(data['seuil'], data['Recall_score'], color='green', lw=2, label='Recall_score')
    plt.plot(data['seuil'], data['F1_score'], color='red', lw=2, label='F1_score')
    #plt.plot(store_score_thresholds['seuil'], store_score_thresholds['Roc_AUC_score'], color='red', lw=2, label='Roc_AUC_score')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Seuil', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.legend(loc="upper left")
    plt.title(f'Score optimized for binary classification obtained with threshold = {optimized_seuil}')
    plt.show()


# %%
def decode_sentiment(score, seuil=0.5):
    if score <= seuil:
        label = 0
    elif score > seuil:
        label = 1
    return label

# %%
#Fonction pour retourner les score des modèles

###############################        ATTENTION : INITIALISATION A PREVOIR AVANT UTILISATION     ################
#Initialisation de la table des résultats
#score_column_names = ["Model Type","Model Name","seuil","F1-Score", "Recall_score", "Precision_score", "Accuracy_score"]
#store_score= pd.DataFrame(columns = score_column_names)

# def evaluation(model,model_name,score_column_names,X_test,y_test, seuil = 0.5, binary_transform=False):
#     # On récupère la prédiction de la valeur positive
#     if binary_transform == True:
#         y_prob = model.predict(X_test)
#         y_pred = y_prob
#     else:
#         y_prob = model.predict_proba(X_test)[:,1]
#         y_pred = np.where(y_prob > seuil, 1, 0)
    
#     # On créé un vecteur de prédiction à partir du vecteur de probabilités
#     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob) # y_prob instead of y_prob #, pos_label=4
#     Roc_AUC_score = auc(false_positive_rate, true_positive_rate)
    
#     F1_score = f1_score(y_test, y_pred)
#     FBeta_score = fbeta_score(y_test, y_pred, average='binary', beta=0.5, pos_label=1) #make_scorer(fbeta_score, beta = 2, pos_label=0 ,average = 'binary')
#     Recall_score = recall_score(y_test, y_pred)
#     Precision_score = precision_score(y_test, y_pred)
#     Accuracy_score = accuracy_score(y_test, y_pred)
    
#     #Plot functions
#     matrix_pred_model(model, model_name, y_test, y_pred) 
#     plot_roc_auc_curve(model_name, false_positive_rate, true_positive_rate)
    
#     score_results = pd.Series([model, model_name, seuil, F1_score, FBeta_score, Recall_score, Precision_score, Accuracy_score, Roc_AUC_score])
#     score_results_stored = pd.DataFrame([score_results.values],  columns = score_column_names)
#     return(score_results_stored)

# def evaluation_to_correct(model,model_name,score_column_names,store_score,X_test,y_test, seuil = 0.5):
#     #Si le seuil n'est pas important
#     y_pred = model.predict(X_test)
#     F1_score = f1_score(y_test, y_pred)#, pos_label=4
#     Recall_score = recall_score(y_test, y_pred) #, pos_label=4
#     Precision_score = precision_score(y_test, y_pred) #, pos_label=4
#     Accuracy_score = accuracy_score(y_test, y_pred)
    
#     matrix_pred_model(model, model_name, y_test,y_pred)  
    
#     #global store_score
#     score_results = pd.Series([model, model_name, seuil, F1_score, Recall_score, Precision_score, Accuracy_score])
#     score_results_stored = pd.DataFrame([score_results.values],  columns = score_column_names)
#     store_score = pd.concat([store_score, score_results_stored], axis=0)
#     return(store_score)

def evaluation(model,model_name,score_column_names,X_test,y_test, seuil = 0.5, binary_predict=False, predict_proba_OK=False):
    # On récupère la prédiction de la valeur positive
    if binary_predict == True:
        y_prob = model.predict(X_test)
        y_pred = y_prob
    elif ((binary_predict == False) and (predict_proba_OK == True)):
        y_prob = model.predict_proba(X_test)[:,1]
        y_pred = np.where(y_prob > seuil, 1, 0)
    elif predict_proba_OK == False:
        y_prob = model.predict(X_test)
        y_pred = np.where(y_prob > seuil, 1, 0)
        y_pred = y_pred.astype(int)
    
    # On créé un vecteur de prédiction à partir du vecteur de probabilités
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob) # y_prob instead of y_prob #, pos_label=4
    Roc_AUC_score = auc(false_positive_rate, true_positive_rate)
    
    F1_score = f1_score(y_test, y_pred)
    FBeta_score = fbeta_score(y_test, y_pred, average='binary', beta=0.5, pos_label=1) #make_scorer(fbeta_score, beta = 2, pos_label=0 ,average = 'binary')
    Recall_score = recall_score(y_test, y_pred)
    Precision_score = precision_score(y_test, y_pred)
    Accuracy_score = accuracy_score(y_test, y_pred)
    
    #Plot functions
    matrix_pred_model(model, model_name, y_test, y_pred) 
    plot_roc_auc_curve(model_name, false_positive_rate, true_positive_rate)
    
    score_results = pd.Series([model, model_name, seuil, F1_score, FBeta_score, Recall_score, Precision_score, Accuracy_score, Roc_AUC_score])
    score_results_stored = pd.DataFrame([score_results.values],  columns = score_column_names)
    return(score_results_stored)

# %%
def plot_model_result(data: pd.DataFrame, score, model_name):
    #Définir la taille du graphique
    fig, ax = plt.subplots(figsize=(15,10))
    #Définir lae type du graphique, ici BARPLOT avec en Labels l'index du nom des libelle
    ax = sns.barplot(data=data, y=model_name, x=score, palette=colors_from_values_float(data[score], "Greens_d"))
    ax.set_xlim((data[score].min() - 0.05), (data[score].max() + 0.02))
    #ax = sns.barplot(data=data, x=model_name, y=score)
    plt.title(f"Score {score} max value is {round(data[score].max(), 2)} computed in model {data.loc[data[score] == data[score].max(), model_name]}")
    #Afficher le graphique
    plt.show()

# %%
def plot_history(history, model_name):
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    x=range(1, len(acc) + 1)
    
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'g', label='Training accuracy')
    plt.plot(x, val_acc, 'c', label='Validation accuracy')
    plt.title(f'Training and Validation Metric: Accuracy')
    plt.legend(loc="upper right")
    
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'g', label='Training loss')
    plt.plot(x, val_loss, 'c', label='Validation loss')
    plt.title(f"Training and Validation loss")
    plt.legend(loc="upper left")

    plt.suptitle(f"Metric & Loss evolution during {model_name} training, (stopped by callback at epochs : {len(acc)})")
    plt.show()


# %% [markdown]
# 


