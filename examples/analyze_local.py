"""SURA - Work at heights evaluation function"""
import logging
from datetime import datetime

import numpy as np

from MLanalyzer.analyzer import analyzer



logging.basicConfig(level=logging.DEBUG)

PERSON_SCALE = 1.12 # Pixel - Real life Object scale
MIN_DIST = 2 # Min distance between persons to be considered near - in meters

# EVALUATION METRICS
# Persons related risks
person = 0.2
no_helmet = 3
person_edge = 2
no_helmet_edge = 6
no_harness_edge = 6

# Obstacles related
obstacle_labels = ['balde', 'carretilla', 'tubo', 'tabla']
obstacle = 0.1
obstacle_edge = 3


predictions_filepath = '/misdoc/datasets/baluarte/from-vrap1/results/predictions'
# predictions_filepath = '/misdoc/datasets/baluarte/analysis-test-2/results/predictions'
# predictions_filepath = '/misdoc/datasets/EAFIT-analysis-test/results/predictions'
# predictions_filepath = '/misdoc/datasets/baluarte/wiw-test/frames-flip/results/predictions'


# date_range = (datetime(2020,8,15).strftime('%s'), datetime(2020,9,15).strftime('%s'))
date_range = None


def label_exists(label, list_objs, id_identifier='id:'):
    """Check if label is in list of objects. If label == 'id'
    Check if label with ID exists on list of objects. and return the ID"""
    if list_objs:
        for l in list_objs:
            obj_label = l['label'].lower()
            if label.lower() == 'id' and obj_label.startswith(id_identifier):
                _id = obj_label.split(':')[-1]
                return _id
            elif obj_label == label.lower():
                return True
    return False


def add_risk_person(_id, risk_label, points ,hist, defaults=None):
    """Add the risk of an _id to hist"""
    if _id:
        if _id not in hist:
            hist[_id] = dict(defaults) if defaults else {}
        hist[_id][risk_label] = points
        if 'total' in hist[_id]:
            hist[_id]['total'] += points
        else:
            hist[_id]['total'] = points

def midpoint(ptA, ptB):
	    return (ptA[0] + ptB[0] * 0.5), (ptB[1])
        

def eval_function(frame_prediction):
    """Sura works at height evaluation function"""
    date = frame_prediction['date']
    # Frame evaluation
    feval = {
        'total':0,
        'sin casco':0,
        'sin casco':0,
        'personas en borde': 0,
        'sin arnés': 0,
        'personas': 0,
        'obstáculos': 0,
        'obstáculos en borde': 0,
    }

    # ID evaluation
    id_feval = {
        'total':0,
        'sin casco':0,
        'sin casco':0,
        'personas en borde': 0,
        'sin arnés': 0,
    }
    ids = {}
    position_ids = {} # persons with ID in frame 'id': (x,y) 
    pixel_ratio = [] # pixel/scale ratio for distance between persons - average
    if 'objects' in frame_prediction:
        for obj in frame_prediction['objects']:
            if obj['label'] == 'persona':
                feval['total'] += person
                feval['personas'] += person

                subobjs = obj['subobject']

                _id = label_exists('id', subobjs)
                if _id: # Person position as middle point of lower edge 
                    geo = obj['boundbox']
                    c1 = [geo['xmin'], geo['ymax']]
                    c2 = [geo['xmax'], geo['ymax']]
                    pos = midpoint(c1, c2)
                    position_ids[_id] = pos
                    if not pixel_ratio: # Calculate pixel/scale ratio
                        pixel_ratio.append((c2[0]-c1[0]) / PERSON_SCALE)

                on_edge = label_exists('cerca de borde', subobjs)
                without_helmet = label_exists('sin casco', subobjs)
                without_harness = label_exists('sin arnes', subobjs)

                if not on_edge and without_helmet:
                    feval['total'] += no_helmet
                    feval['sin casco'] += no_helmet
                    add_risk_person(_id, 'sin casco', no_helmet, ids, defaults=id_feval)
                    
                if on_edge:
                    feval['total'] += person_edge
                    feval['personas en borde'] += person_edge

                    if without_helmet:
                        feval['total'] += no_helmet_edge
                        feval['sin casco'] += no_helmet_edge
                        add_risk_person(_id, 'sin casco', no_helmet_edge, ids, defaults=id_feval)

                    if without_harness:
                        feval['total'] += no_harness_edge
                        feval['sin arnés'] += no_harness_edge
                        add_risk_person(_id, 'sin arnés', no_harness_edge, ids, defaults=id_feval)

            elif obj['label'] in obstacle_labels:
                feval['total'] += obstacle
                feval['obstáculos'] += obstacle

                if 'subobject' in obj:
                    subobjs = obj['subobject']
                    on_edge = label_exists('cerca de borde', subobjs)
                    if on_edge:
                        feval['total'] += obstacle_edge
                        feval['obstáculos en borde'] += obstacle_edge
    # Add personas-cerca to each id
    for i in ids:
        ids[i]['personas-cerca'] = []
    
    
    # Calculate distance between persons with ID
    if position_ids:
        ratio = np.average(pixel_ratio)
        for _id, pos in position_ids.items():
            for next_id, next_pos in position_ids.items():
                if _id != next_id:
                    # Calculate distance between persons
                    p1 = np.array(pos)
                    p2 = np.array(next_pos)
                    dist = np.linalg.norm(p1 - p2) / ratio
                    if dist < MIN_DIST:
                        ids[_id]['personas-cerca'].append(next_id)
    return date, feval, ids








def eval_function_detailed(frame_prediction):
    """Sura works at height evaluation function"""
    date = frame_prediction['date']
    # Frame evaluation
    feval = {
        'total':0,
        'sin casco':0,
        'sin casco':0,
        'personas en borde': 0,
        'sin arnés': 0,
        'personas': 0,
        'balde en borde': 0,
        'carretilla en borde': 0,
        'tubo en borde': 0,
        'tabla en borde': 0,
        'obstáculos': 0,
        # 'obstáculos en borde': 0,
    }

    # ID evaluation
    id_feval = {
        'total':0,
        'sin casco':0,
        'sin casco':0,
        'personas en borde': 0,
        'sin arnés': 0,
    }
    ids = {}
    position_ids = {} # persons with ID in frame 'id': (x,y) 
    pixel_ratio = [] # pixel/scale ratio for distance between persons - average
    if 'objects' in frame_prediction:
        for obj in frame_prediction['objects']:
            if obj['label'] == 'persona':
                feval['total'] += person
                feval['personas'] += person

                subobjs = obj['subobject']

                _id = label_exists('id', subobjs)
                if _id: # Person position as middle point of lower edge 
                    geo = obj['boundbox']
                    c1 = [geo['xmin'], geo['ymax']]
                    c2 = [geo['xmax'], geo['ymax']]
                    pos = midpoint(c1, c2)
                    position_ids[_id] = pos
                    if not pixel_ratio: # Calculate pixel/scale ratio
                        pixel_ratio.append((c2[0]-c1[0]) / PERSON_SCALE)

                on_edge = label_exists('cerca de borde', subobjs)
                without_helmet = label_exists('sin casco', subobjs)
                without_harness = label_exists('sin arnes', subobjs)

                if not on_edge and without_helmet:
                    feval['total'] += no_helmet
                    feval['sin casco'] += no_helmet
                    add_risk_person(_id, 'sin casco', no_helmet, ids, defaults=id_feval)
                    
                if on_edge:
                    feval['total'] += person_edge
                    feval['personas en borde'] += person_edge

                    if without_helmet:
                        feval['total'] += no_helmet_edge
                        feval['sin casco'] += no_helmet_edge
                        add_risk_person(_id, 'sin casco', no_helmet_edge, ids, defaults=id_feval)

                    if without_harness:
                        feval['total'] += no_harness_edge
                        feval['sin arnés'] += no_harness_edge
                        add_risk_person(_id, 'sin arnés', no_harness_edge, ids, defaults=id_feval)

            elif obj['label'] in obstacle_labels:
                feval['total'] += obstacle
                feval['obstáculos'] += obstacle

                if 'subobject' in obj:
                    subobjs = obj['subobject']
                    on_edge = label_exists('cerca de borde', subobjs)
                    if on_edge:
                        feval['total'] += obstacle_edge
                        feval[f'{obj["label"]} en borde'] += obstacle_edge
    # Add personas-cerca to each id
    for i in ids:
        ids[i]['personas-cerca'] = []
    
    
    # Calculate distance between persons with ID
    if position_ids:
        ratio = np.average(pixel_ratio)
        for _id, pos in position_ids.items():
            for next_id, next_pos in position_ids.items():
                if _id != next_id:
                    # Calculate distance between persons
                    p1 = np.array(pos)
                    p2 = np.array(next_pos)
                    dist = np.linalg.norm(p1 - p2) / ratio
                    if dist < MIN_DIST:
                        ids[_id]['personas-cerca'].append(next_id)
    return date, feval, ids
















analyzer(predictions_filepath, mode='analyze',eval_function=eval_function_detailed,  similarity=0.6, date_range=date_range)