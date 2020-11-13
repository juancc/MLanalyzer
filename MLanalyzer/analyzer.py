"""
MLanalyzer
Analyze and display information of models predictions of a time based dataset
Modes:
    - Predict: predict images and save annotation file
    - Analyze: read annotation file and make data analisys

JCA
Vaico
"""
from os import path, makedirs
from MLanalyzer.auxfunc.modes import predict, analize

def analyzer(filepath, model=None, date_splitter=None, mode=None, eval_function=None, 
    saving_condition=None, splits=20, similarity=0.05):
    print(f'Running Analyzer. Mode: {mode if mode else "Complete"}')
    res = None
    if mode == 'predict' or not mode:
        print(f'Making predictions on {filepath}')
        print(' - Creating Results folder...')
        makedirs(path.join(filepath, 'results'), exist_ok=True)
        args = [filepath, model]
        kwargs={} 
        if date_splitter: kwargs['date_splitter'] = date_splitter
        if saving_condition: kwargs['saving_condition'] = saving_condition
        
        ann_path = predict(*args, **kwargs)
        if not mode:
            mode = 'analyze'
            filepath = ann_path
        else:
            res = 'ok'
    
    if mode == 'analyze':
        print(f'Making analysis on {filepath}')
        res = analize(filepath, eval_function, splits=splits, similarity=similarity)
    if res:
        print('Done')