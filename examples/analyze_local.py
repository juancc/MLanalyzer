from MLanalyzer.analyzer import analyzer

predictions_filepath = '/misdoc/datasets/baluarte/analysis-test/predictions'
config = {
    'persona':{
        'val': 1,
        'subobjects':{
            'sin casco':{
                'val': 3
            }
        }
    }
}


analyzer(predictions_filepath, mode='analyze', predictions_config=config)