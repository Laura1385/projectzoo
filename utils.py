import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import squarify 
import sklearn
import scipy as sp
import warnings
warnings.filterwarnings("ignore")

def border_msg(msg):
    row = len(msg)
    h = ''.join(['+'] + ['-' *row] + ['+'])
    result= h + '\n'"|"+msg+"|" '\n' + h
    print(result)

def analisys(filename):
    df = pd.read_csv(filename,sep=',')
    border_msg('Nome File:')
    print(filename, "\n")

    border_msg('Informazioni DataFrame:')
    df.info()

    print("\n - Numero di osservazioni (righe) x caratteristiche (colonne):", df.shape, "\n\n")
    print(border_msg('Anteprima del DataFrame:'),df.head(5), "\n")
    print(border_msg('Statistiche di base:'),df.describe(), "\n")
    print(border_msg('Analisi variabili:'), df.nunique(), "\n\n")

    dfname= str(filename) 
    name= dfname[:-4]#take values before point
    prefix= "df_"
    f = print("Rinomino il DataFrame in:", prefix + name)

def confusion_matrix(df):
    #Import data
    data = df
    corr = data.corr() #osx need (numeric_only=True)

    #Print data
    #print(data.corr())

    #Create space of the graphic 
    fig, ax = plt.subplots(figsize=(9, 6))

    #Create the mask to NOT have graphic redundancy
    mask = np.triu(np.ones_like(corr, dtype=bool), k = 1)

    #Create correlation palette and heat map
    cmap = sns.color_palette("Spectral_r", as_cmap=True)
    dataplot = sns.heatmap(corr, mask=mask, center=0, annot=True, annot_kws={'size': 8}, fmt='.1f', cmap=cmap, linewidths=.35)
    plt.title(f"Correlation Matrix Animals", fontsize=15)
    plt.show()
    
def update_animal_name(df, old_name, new_name):
    index = df.index[df['animal_name (nome_animale)'].str.startswith(old_name)]
    if len(index) > 0:
        old_value = df.at[index[0], 'animal_name (nome_animale)']
        new_full_name = old_value.replace(old_name, new_name)
        df.at[index[0], 'animal_name (nome_animale)'] = new_full_name
        print(f"Ho modificato '{old_value}' con '{new_full_name}'")
    else:
        print(f"Nessun animale trovato che inizi con '{old_name}'")

def modify_column(df, animal_column, column_to_modify, values_to_modify):
    for animal, value in values_to_modify.items():
        # values_to_modify is a dictionary with animal's name like keys to modify and their new values like value
        df.loc[df[animal_column] == animal, column_to_modify] = value
        print("Fatto!")


