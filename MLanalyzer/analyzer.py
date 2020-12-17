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
import logging

from MLanalyzer.auxfunc.modes import predict, analize

logger = logging.getLogger(__name__)


def analyzer(filepath, model=None, date_splitter=None, mode=None, eval_function=None, 
    saving_condition=None, splits=20, similarity=0.05, date_range=None, draw_formats=None):
    logger.info(f'Running Analyzer. Mode: {mode if mode else "Complete"}')
    res = None
    if mode == 'predict' or not mode:
        logger.info(f'Making predictions on {filepath}')
        logger.info(' - Creating Results folder...')
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
        logger.info(f'Making analysis on {filepath}')
        # Analyze responses
        res = analize(filepath, eval_function, splits=splits, similarity=similarity, date_range=date_range, draw_formats=draw_formats)
    if res:
        logger.info('Done')