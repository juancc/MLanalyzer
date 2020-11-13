"""
Analyzer steps:
    - predict: predict images and save annotation file
    - analyze: read annotation file and make data analisys
"""
from os import path, environ, listdir, makedirs
import json
from datetime import datetime
import itertools
from random import random

from tqdm import tqdm
import cv2 as cv
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')


from MLanalyzer.auxfunc.date_splitters import nvr_default_1
from MLanalyzer.auxfunc.general import batch


def predict(dataset_path, model, date_splitter=nvr_default_1, saving_condition=lambda obj: True):
    """Make and store predictions using model
        :param im_path: (str) path to images
        :param model: (MLinference) prediction model: or a function that have predict()
        :param date_splitter: (func) Function that returns the date from epoch based the filename
        :param saving_condition: (func) Function hat return true if prediction objects sould be saved
    """
    todo = tqdm(listdir(dataset_path))
    ann_path = path.join(dataset_path, 'results' ,'predictions.json')

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



def do_analysis(res, date_epoch, savepath, name=None, show=False):
    """Perform the data analysis and plotting based the responses and dates. 
    :param name: (str) for save results in a specific folder
    :param show: (bool) display plots
    :returns: list of total and sorted dates
    """
    if name:
        # Saving results for ids
        savepath = path.join(savepath, name)
        makedirs(savepath, exist_ok=True)
    else:
        name = 'general'
    # Sort by dates
    dates = []
    reval = {'total':[]}
    sorted_dates = list(res)
    sorted_dates.sort()

    # Remove doubles
    date_epoch = list(set(date_epoch))
    date_epoch.sort()

    for d in sorted_dates:
        dates.append(d)
        if d in res: update_results(res[d], reval)
    
    # Analysis metrics
    total_sum = sum(reval['total'])
    # Time behaviour plot 
    fig = plt.figure(figsize=(18, 6))
    axes = fig.add_subplot(111)
    plt.title(f'Contribución en el tiempo de {name}')
    # Polar plots
    categories_values = [] # porcentual of total
    categories = [] 
    results = f'----- Results of {name}'
    for k,v in reval.items():
        if isinstance(v[0], list): # If results are a list
            v = [len(i) for i in v]
        eval_average = np.average(v)
        eval_std = np.std(v)
        max_val = np.amax(v)
        max_idx = np.where(v == np.amax(v))[0][0]
        max_date = dates[max_idx]
        _sum = np.sum(v)
        results = f'{results}----\n * {k}\n   - Total {_sum}\n   - Average {eval_average}\n   - STD: {eval_std}\n   - Max val: {max_val} in {max_date}'

        # Polar plot
        sum_v = sum(v)
        if k !='total': 
            categories.append(k)
            categories_values.append(10*sum_v/total_sum)

    # Total Data
    total = list(reval['total'])

    # Polynomial Fit
    polynomial_coeff = np.polyfit(date_epoch, total, 1)
    fit_result = f'Total Curve Fit \n - Polynomial Coefficients: {polynomial_coeff}'
    results = f'{results}\n{fit_result}'
    print(fit_result)

    savefile = path.join(savepath, 'analysis_results.txt')
    print(f'Saving results {savefile}')
    with open(savefile, 'w') as f:
        f.write(results)

    # Display and save plots
    
    # Time behaviour
    # Fix data for time plot
    reval.pop('total')
    # Convert responses that are lists to lens
    list_responses = {}
    for k,v in reval.items():
        if isinstance(v[0], list): # If results are a list
            list_responses[k] = v
            reval[k] = [len(i) for i in v]  

    plt.stackplot(dates, reval.values(),
             labels=reval.keys())

    plt.axhline(y=np.average(total), xmin=0, xmax=50, linestyle='--', color=(1,0,0.5), label='Promedio')
    plt.gcf().autofmt_xdate()

    # List responses
    if list_responses:
        for metric, values in list_responses.items():
            all_vals = list(itertools.chain.from_iterable(values))

            unique, frequency = np.unique(all_vals, return_counts = True) 
            fig_list = plt.figure()
            plt.xlabel(f'ID de {metric}' )
            plt.ylabel("Número de veces") 

            plt.bar(unique,frequency)
            fig_list.savefig(path.join(savepath,f'id-{metric}'))


    # Draw Fit
    ynew = np.poly1d(polynomial_coeff)
    plt.plot(dates,ynew(date_epoch), label='Tendencia', color=(0,0,0), )

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
    plt.title(f"Evaluación de categorías de {name}")

    fig_2.savefig(path.join(savepath, 'categories_eval.png'))

    # Shot plots
    if show: plt.show()
    return date_epoch, total

def is_similar(coeff1, coeff2, thresh):
    """ Return if coefficient are similar. If difference is under thresh percentual"""
    if len(coeff1) != len(coeff2):
        return False
    for i in range(len(coeff1)):
        if coeff1[i]>0 == coeff2[i]>0:# Different signs
            return False
        tot = abs(coeff1[i]) + abs(coeff2[i])
        dif = abs(coeff2[i]) - abs(coeff1[i])
        if dif/tot >thresh:
            return False
    return True



def clustering(res, dates, splits, similarity, name='Total'):
    """Perform a clustering analysis based the slope of the splits. Dataset will be splitted in the split number
    A linear approximation will be made on each group. If is under similar thresh and with
    same sign will be considered the same group"""
    print(f'Performing clustering with: {splits} splits and similarity of: {similarity}')
    n = len(res)//splits

    # Draw Total
    fig = plt.figure(figsize=(20, 6))
    axes = fig.add_subplot(111)
    plt.title(f'Agrupación ({name})')
    plt.plot(dates, res, label='Total', color=(0.5,0.5,0.5), linestyle='--')

    batch_n = 0
    # Batches that contain current group
    curr_res = [] 
    curr_dates = []
    cluster_res = []
    cluster_dates = []
    for _date, _res in zip(batch(dates, n), batch(res, n)):
        if not curr_res:
            curr_res.extend(_res)
            curr_dates.extend(_date)
            continue
        # Perform linear fit of current batch [m,b]
        curr_coeff = np.polyfit(curr_dates, curr_res, 1)
        
        # Perform linear fit of new batch
        new_coeff = np.polyfit(_date, _res, 1)
        
        if is_similar(curr_coeff, new_coeff, similarity):
            curr_res.extend(_res)
            curr_dates.extend(_date)
            continue
        
        # Not similar
        cluster_res.append(curr_res)
        cluster_dates.append(curr_dates)

        # Draw cluster
        ynew = np.poly1d(curr_coeff)
        plt.fill_between(curr_dates, ynew(curr_dates), label=batch_n, color=(random(),random(),random()), )

        # Update current
        curr_res = _res
        curr_dates = _date
        batch_n += 1
    print(f' - Total of clusters: {len(cluster_res)}')




    # axes.legend()
    plt.show()
    
    
    
    
    exit()

def analize(annotation_path, eval_function, splits=20, similarity=0.05):
    """Analize predictions from annotation file
        :param annotation_path: (str) filepath to json-lines file with predictions
        :param eval_function: (func) function that recieve
            the predictions of a date and return: date, evaluation number 
            or a dict and the ID evaluation if ids are present
        :param splits: (int) number of batches to split the data to make linear grouping. 
            Total risk will be splitted and join the parts with similar splope
        _param similarity: (float) percentage of difference to be considered same group
    """
    # Add a date and the evaluation according to the prediction configurarion
    res = {} # for each 'date':{'total': int, 'other_objects_eval': int} 
    date_epoch = [] # date in seg. For linear analysis

    # For IDS
    id_res = {} # id:{date1: {items of eval}, date2: {items of eval}}
    id_date_epoch = {} # id:date_epoch
    savepath, _ = path.split(annotation_path)
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    for l in lines:
        l = json.loads(l)

        f_date, f_eval, ids = eval_function(l)
        if f_date:
            date_epoch.append(f_date)
            timedate = datetime.fromtimestamp(f_date)
            if isinstance(f_eval, dict):
                res[timedate] = f_eval
            else:
                res[timedate] = {'total': f_eval}
            
            if ids:
                for _id, id_eval in ids.items():
                    # Add result
                    if _id in id_res:
                        id_res[_id][timedate] =  id_eval
                    else:
                        id_res[_id] = {timedate: id_eval}
                    # Add date
                    if _id in id_date_epoch:
                        id_date_epoch[_id].append(f_date)
                    else:
                        id_date_epoch[_id] = [f_date]
    
    # Perform general resuls  
    dates, total = do_analysis(res, date_epoch, savepath, show=False)

    # Perform clustering on general
    clustering(total, dates, splits, similarity)


    # Perform IDs results
    for _id, _res in id_res.items():
        do_analysis(_res, id_date_epoch[_id], savepath,name=_id, show=False)   


    return savepath