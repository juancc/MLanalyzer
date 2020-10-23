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

def update_results(feval, hist):
    """Update evaluation results of analysis funcion when is a dict
        :param feval: (dict) result of evaluation function
        :param hist: (dict) Previous results
    """
    for k,v in feval.items():
        if k in hist:
            hist[k].append(v)
        else:
            hist[k] = [v]

def analize(annotation_path, eval_function):
    """Analize predictions from annotation file
        :param annotation_path: (str) filepath to json-lines file with predictions
        :param eval_function: (func) function that recieve
            the predictions of a date and return date and evaluation number 
            or a dict with the total and other evaluated variables
    """
    # Add a date and the evaluation according to the prediction configurarion
    dates = []
    reval = {'total':[]}
    
    savepath, _ = path.split(annotation_path)

    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    for l in lines:
        l = json.loads(l)
        try:
            f_date, f_eval = eval_function(l)
            timedate = datetime.fromtimestamp(f_date)
            dates.append(timedate)

            if isinstance(f_eval, dict):
                update_results(f_eval, reval)
            else:
                reval['total'].append(f_eval)
        except TypeError as e:
            print(f'\n Error: {e}. Most provide a valid an evaluation function for analysis\n')
            return None

    # Analysis metrics
    total_sum = sum(reval['total'])
    # Time behaviour plot 
    fig = plt.figure(figsize=(18, 6))
    axes = fig.add_subplot(111)
    plt.title('Contribución en el tiempo')
    # Polar plots
    categories_values = [] # porcentual of total
    categories = [] 

    results = 'Results\n'
    for k,v in reval.items():
        # Results
        eval_average = np.average(v)
        eval_std = np.std(v)
        max_val = np.amax(v)
        max_idx = np.where(v == np.amax(v))[0][0]
        max_date = dates[max_idx]
        results = f'{results}----\n {k}\n - Average {eval_average}\n - STD: {eval_std}\n - Max val: {max_val} in {max_date}'

        # Add plots time
        # plt.bar(dates, v, label=k)

        # Polar plot
        sum_v = sum(v)
        if k !='total': 
            categories.append(k)
            categories_values.append(10*sum_v/total_sum)
    print(results)

    savefile = path.join(savepath, 'analysis_results.txt')
    print(f'Saving results {savefile}')
    with open(savefile, 'w') as f:
        f.write(results)

    # Display and save plots
    # Time behaviour
    reval.pop('total')
    plt.stackplot(dates, reval.values(),
             labels=reval.keys())

    plt.gcf().autofmt_xdate()
    axes.legend()
    fig.savefig(path.join(savepath, 'time-eval.png'))

    # Polar plots
    categories_values.append(categories_values[0])# complete de circle
    fig_2 = plt.figure(figsize=(10, 6))
    plt.subplot(polar=True)
 
    theta = np.linspace(0, 2 * np.pi, len(categories_values))

    # Arrange the grid into equal parts in degrees
    lines, labels = plt.thetagrids(range(0, 360, int(360/len(categories))), (categories))

    # Plot actual sales graph
    plt.plot(theta, categories_values, label='Total')
    plt.fill(theta, categories_values, 'b', alpha=0.1)

    # Add legend and title for the plot
    plt.legend()
    plt.title("Evaluación de categorías")

    fig_2.savefig(path.join(savepath, 'categories_eval.png'))

    # Shot plots
    plt.show()

    return savepath