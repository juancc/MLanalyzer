"""
MLanalyzer
Analyze and display information of models predictions of a time based dataset
Modes:
    - Predict: predict images and save annotation file
    - Analyze: read annotation file and make data analisys

JCA
Vaico
"""
from MLanalyzer.auxfunc.modes import predict, analize

def analyzer(filepath, model=None, date_splitter=None, mode=None, predictions_config=None, saving_condition=None):
    print(f'Running Analyzer. Mode: {mode if mode else "Complete"}')
    if mode == 'predict' or not mode:
        print(f'Making predictions on {filepath}')

        args = [filepath, model]
        kwargs={} 
        if date_splitter: kwargs['date_splitter'] = date_splitter
        if saving_condition: kwargs['saving_condition'] = saving_condition
        
        ann_path = predict(*args, **kwargs)
        if not mode:
            mode = 'analyze'
            filepath = ann_path
    
    if mode == 'analyze':
        print(f'Making analysis on {ann_path}')
        analize(filepath, predictions_config)
    print('Done')