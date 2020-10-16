"""
Analyzer steps:
    - predict: predict images and save annotation file
    - analyze: read annotation file and make data analisys
"""
from os import path, environ, listdir
import json
from datetime import datetime

from tqdm import tqdm
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from MLanalyzer.auxfunc.date_splitters import nvr_default_1


def predict(dataset_path, model, date_splitter=nvr_default_1, saving_condition=lambda obj: True):
    """Make and store predictions using model
        :param im_path: (str) path to images
        :param model: (MLinference) prediction model: or a function that have predict()
        :param date_splitter: (func) Function that returns the date from epoch based the filename
        :param saving_condition: (func) Function hat return true if prediction objects sould be saved
    """
    todo = tqdm(listdir(dataset_path))
    ann_path = path.join(dataset_path, 'predictions.json')

    print(f' - Predicting images from:{dataset_path}')
    print(f' - Saving predictions in in {ann_path}')
    with open(ann_path, "w") as handler:
        for filename in todo:
            full_path = path.join(dataset_path, filename)
            try:
                im = cv.imread(full_path)
                objs = model.predict(im, model)
                date = date_splitter(filename)

                line = {
                    'objects': [obj._asdict() for obj in objs if saving_condition(obj)],
                    'frame_id': full_path,
                    'date': date
                }
                handler.write(json.dumps(line) + '\n')
            except Exception as e:
                print(f'Error: {e} on frame:{full_path}')
    return ann_path

def analize(annotation_path, eval_function):
    """Analize predictions from annotation file
        :param annotation_path: (str) filepath to json-lines file with predictions
        :param eval_function: (func) function that recieve the predictions of a date and return date and evaluation number
    """
    # Add a date and the evaluation according to the prediction configurarion
    dates = []
    reval = []
    
    savepath, _ = path.split(annotation_path)

    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    for l in lines:
        l = json.loads(l)
        try:
            f_date, f_eval = eval_function(l)
            timedate = datetime.fromtimestamp(f_date)
            dates.append(timedate)
            reval.append(f_eval)
        except TypeError as e:
            print('\n Error: Most provide an evaluation function for analysis\n')
            return None

    # Analysis metrics
    eval_average = np.average(reval)
    eval_std = np.std(reval)
    max_val = np.amax(reval)
    max_idx = np.where(reval == np.amax(reval))[0][0]
    max_date = dates[max_idx]

    results = f'Results\n - Average {eval_average}\n - STD: {eval_std}\n - Max val: {max_val} in {max_date}'
    print(results)

    savefile = path.join(savepath, 'analysis_results.txt')
    print(f'Saving results {savefile}')
    with open(savefile, 'w') as f:
        f.write(results)

    # Time behaviour plot   
    fig = plt.figure()
    axes = fig.add_subplot(111)
    plt.title('Evaluation on time')
    plt.plot(dates, reval)
    plt.gcf().autofmt_xdate()
    fig.savefig(path.join(savepath, 'time-eval.png'))
    plt.show()

    return savepath