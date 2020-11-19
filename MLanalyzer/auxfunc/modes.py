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
import logging

from tqdm import tqdm
import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from MLanalyzer.auxfunc.date_splitters import nvr_default_1
from MLanalyzer.auxfunc.general import batch

plt.style.use('seaborn')
logger = logging.getLogger(__name__)

DAYS = [31,28,31,30,31,30,31,31,30,31,30,31]


def predict(dataset_path, model, date_splitter=nvr_default_1, saving_condition=lambda obj: True):
    """Make and store predictions using model
        :param im_path: (str) path to images
        :param model: (MLinference) prediction model: or a function that have predict()
        :param date_splitter: (func) Function that returns the date from epoch based the filename
        :param saving_condition: (func) Function hat return true if prediction objects sould be saved
    """
    todo = tqdm(listdir(dataset_path))
    ann_path = path.join(dataset_path, 'results' ,'predictions.json')

    logger.info(f' - Predicting images from:{dataset_path}')
    logger.info(f' - Saving predictions in in {ann_path}')
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
                logger.error(f'Error: {e} on frame:{full_path}')
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



def do_analysis(res, date_epoch, savepath, name=None, show=False, outlier_sigma=2):
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

    for d in sorted_dates:
        dates.append(d)
        if d in res: update_results(res[d], reval)
    
    # Analysis metrics
    total_sum = sum(reval['total'])

    # Outliers: data greater than sigma
    outlier_ref = np.mean(reval['total']) + outlier_sigma*np.std(reval['total'])
    total_arr = np.array(reval['total'])
    outliers_idx = np.argwhere(total_arr>outlier_ref)
    outliers_date = [dates[i[0]] for i in outliers_idx]
    outliers_vals = [reval['total'][i[0]] for i in outliers_idx]

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
    logger.info(fit_result)

    savefile = path.join(savepath, 'analysis_results.txt')
    logger.info(f'Saving results {savefile}')
    with open(savefile, 'w') as f:
        f.write(results)

    # Display and save plots
    
    # Time behaviour
    fig = plt.figure(figsize=(20, 8))
    axes = fig.add_subplot(111)
    plt.title(f'Contribución en el tiempo de {name}')

    # Plot outliers
    plt.scatter(outliers_date, outliers_vals, label='Outliers', marker='.', color=(1,0,0))
    plt.axhline(y=outlier_ref, xmin=0, xmax=50, linestyle='--', color=(0.5,0.5,0.5), label=f'Sigma {outlier_sigma}')


    # Number of observations by day
    logger.info('Saving observations by date')
    # Fill dates with 0
    num_obs_date = {}

    Y = dates[0].year
    for m in range(dates[0].month, dates[-1].month+1):
        n_days = DAYS[m-1]
        for dd in range(n_days):
            key = datetime(Y, m, dd+1)
            num_obs_date[key] = 0

    for d in dates:
        Y = d.year
        m = d.month
        dd = d.day
        key = datetime(Y, m, dd)
        num_obs_date[key] += 1

    plt.plot(num_obs_date.keys(), num_obs_date.values(), label='Número de observaciones', color=(1,0,0))

    # Fix data for time plot
    reval.pop('total')
    # Convert responses that are lists to lens
    list_responses = {}
    for k,v in reval.items():
        if isinstance(v[0], list): # If results are a list
            list_responses[k] = v
            reval[k] = [len(i) for i in v]  
    # Set colors of the elements of the evaluation
    eval_colors = [(random(),random(),random()) for i in reval ]
    plt.stackplot(dates, reval.values(),
             labels=reval.keys(), 
             colors=eval_colors,
             )

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

    reval['total'] = total
    return dates, reval

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



def clustering(reval, epoch_dates, splits, similarity, savepath, grouping='total'):
    """Perform a clustering analysis based the slope of the splits. Dataset will be splitted in the split number
    A linear approximation will be made on each group. If is under similar thresh and with
    same sign will be considered the same group"""
    logger.info(f'Performing clustering with: {splits} splits and similarity of: {similarity}')
    logger.info(f' - Grouping by {grouping}')
    res = reval[grouping]
    total_data = len(res)
    n = total_data//splits# number of data by split

    savepath = path.join(savepath, 'clusters')
    logger.info(f' - Making directory for cluster results at {savepath}')
    makedirs(savepath, exist_ok=True)

    # Draw Total
    fig = plt.figure(figsize=(20, 6))
    axes = fig.add_subplot(111)
    plt.title(f'Agrupación ({grouping})')

    batch_n = 0
    # Batches that contain current group
    # curr_res = [] 
    # curr_dates = []
    # cluster_res = []
    # cluster_dates = []

    clusters = [] #(init,end)
    cluster_init = 0
    cluster_end = min(total_data, n)

    for s in range(splits-1):
        init = n*(s+1)
        end = min(total_data, init+n)

        curr_coeff = np.polyfit(epoch_dates[cluster_init:cluster_end], res[cluster_init:cluster_end], 1)
        
        # Perform linear fit of new batch
        new_coeff = np.polyfit(epoch_dates[init:end], res[init:end], 1)
        
        if is_similar(curr_coeff, new_coeff, similarity):
            cluster_end = end
            continue
        
        # Not similar
        # Draw cluster
        ynew = np.poly1d(curr_coeff)
        dates = [datetime.fromtimestamp(stamp) for stamp in epoch_dates[cluster_init:cluster_end]]
        plt.plot(dates, ynew(epoch_dates[cluster_init:cluster_end]), color=(0,0,0),  linestyle='--')
        plt.fill_between(dates, res[cluster_init:cluster_end],  color=(random(),random(),random()), label=batch_n)

        # Update cluster
        clusters.append((cluster_init,cluster_end))
        cluster_init = init
        cluster_end = end
        batch_n += 1
    logger.info(f' - Total of clusters: {len(clusters)}')
    axes.legend()
    fig.savefig(path.join(savepath, f'clustering-{grouping}.png'))

    results = f'Results\n  Number of clusters: {len(clusters)}\n'

    logger.info(' - Plotting cluster metrics composition')
    # Plot data of other metrics
    cluster_name = 0
    for i,e in clusters:
        metric_sum = []
        metric_labels = []
        results = results + f'--- Cluster {cluster_name} --- \n'
        total = 0
        for k,v in reval.items():
            tot_metric = sum(v[i:e])
            if tot_metric and k != 'total':
                total += tot_metric
                metric_labels.append(k)
                metric_sum.append(tot_metric)
                results = results + f' - {k}: {tot_metric}\n'

        results = results + f' * Total: {total}\n'

        fig_metrics = plt.figure()
        plt.title(f'Composición de cluster {cluster_name}')

        plt.pie(metric_sum, labels=metric_labels,)
        fig_metrics.savefig(path.join(savepath, f'composition-{cluster_name}.png'))
        cluster_name += 1

    logger.info(' - Saving results')
    with open(path.join(savepath,'results.txt'), 'w') as f:
        f.write(results)


def analize(annotation_path, eval_function, splits=20, similarity=0.05, date_range=None):
    """Analize predictions from annotation file
        :param annotation_path: (str) filepath to json-lines file with predictions
        :param eval_function: (func) function that recieve
            the predictions of a date and return: date, evaluation number 
            or a dict and the ID evaluation if ids are present
        :param splits: (int) number of batches to split the data to make linear grouping. 
            Total risk will be splitted and join the parts with similar splope
        _param similarity: (float) percentage of difference to be considered same group
        :param date_range: (tuple) (initial, end) on seconds since epoch
    """
    if date_range:
        logger.info(f'Using a range of dates. From {date_range[0]} to {date_range[1]}')

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
            if date_range:
                if not (f_date>float(date_range[0]) and f_date<float(date_range[1])):
                    continue
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
    
    # Remove doubles
    date_epoch = list(set(date_epoch))
    date_epoch.sort()

    # Perform general resuls  
    dates, reval = do_analysis(res, date_epoch, savepath, show=False)

    # Perform clustering on general
    clustering(reval, date_epoch, splits, similarity, savepath)

    # Perform IDs results
    if id_res: logger.info('Doing analysis by IDS')
    for _id, _res in id_res.items():
        do_analysis(_res, id_date_epoch[_id], savepath,name=_id, show=False)   


    return savepath