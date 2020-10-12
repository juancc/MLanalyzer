"""
MLanalyzer
Analyze and display information of models predictions of a time based dataset

JCA
Vaico
"""
from os import listdir
from os import path, environ
import json

from tqdm import tqdm
import cv2 as cv

def predict(dataset_path, model, date_splitter):
    """Make and store predictions using model
        :param im_path: (str) path to images
        :param model: (MLinference) prediction model: or a function that have predict()
        :param date_splitter: (func) Function that returns a dict with {year, month, day, hour, second} 
            if not present will be taken from year to seconds the available one
    """
    todo = tqdm(listdir(dataset_path))
    ann_path = path.join(dataset_path, 'annotation.json')

    print(' - Predicting images in:{}'.format(dataset_path))
    print(' - Saving labels in {}'.format(ann_path))
    with open(ann_path, "w") as handler:
        i=0
        for f in todo:
            full_path = path.join(dataset_path, f)
            try:
                im = cv.imread(full_path)
                objs = model.predict(im, model)
                
                date = f.split()

                res = {
                    'objects': objs,
                    'frame_id': full_path,
                    'date': None
                }