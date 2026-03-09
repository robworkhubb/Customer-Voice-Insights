import pandas as pd
import numpy as np

def preprocessing(filepath):
    # Score e text sono le colonne che più ci interessa trovare il pattern
    df = pd.read_csv(filepath, usecols=['Score', 'Text'])
    df.dropna(inplace=True)
    
    # All'azienda fittizia per cui lavoriamo interessa solo se ci sono recensioni 1-2 stelle(quindi se abbandonano) oppure 4-5 (se continueranno ad usare il servizio) quindi escludiamo 3 dallo score
    df = df[df['Score'] != 3].copy()
    
    # Ora mappiamo: 4,5 classe 1 | 1,2 classe 0
    df['Sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)
    
    return df['Text'].values, df['Sentiment'].values