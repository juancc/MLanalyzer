from MLanalyzer.analyzer import analyzer

predictions_filepath = '/misdoc/datasets/baluarte/analysis-test/predictions'


def eval_function(frame_prediction):
    person = 1
    no_helmet = 3
    person_edge = 2
    no_helmet_edge = 6
    no_harness_edge = 6

    date = frame_prediction['date']
    feval = {
        'total':0,
        'sin casco':0,
        'cerca de borde': 0,
        'sin arnés': 0
    
    }

    def label_exists(label, list_objs):
        for l in list_objs:
            if l['label'].lower() == label.lower():
                return True
        return False

    if 'objects' in frame_prediction:
        for obj in frame_prediction['objects']:
            if obj['label'] == 'persona':
                feval['total'] += person
                subobjs = obj['subobject']
                
                on_edge = label_exists('cerca', subobjs)
                without_helmet = label_exists('sin casco', subobjs)
                without_harness = label_exists('sin arnes', subobjs)

                if not on_edge and without_helmet:
                    feval['total'] += no_helmet
                    feval['Sin casco'] += no_helmet
                if on_edge:
                    feval['total'] += person_edge
                    feval['cerca de borde'] += person_edge

                    if without_helmet:
                        feval['total'] += no_helmet_edge
                        feval['Sin casco'] += no_helmet_edge
                    if without_harness:
                        feval['total'] += no_harness_edge
                        feval['sin arnés'] += no_harness_edge
    return date, feval

analyzer(predictions_filepath, mode='analyze',eval_function=eval_function)