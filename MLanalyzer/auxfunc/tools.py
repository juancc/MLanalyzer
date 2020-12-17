from os import path, environ, listdir, makedirs
from datetime import datetime
import logging
from random import random

# import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from MLanalyzer.auxfunc.date_splitters import nvr_default_1

# Days of the month
days = [31,28,31,30,31,30,31,31,30,31,30,31]
months_names = ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']

plt.style.use('seaborn')
logger = logging.getLogger(__name__)

def analyze_observation_dates(dataset_path, date_splitter=nvr_default_1, by='day', save=False, savepath=None, show=True):
    """Plot the distribution of the images dates"""
    logger.info(f'Analyzing dates on: {dataset_path}')
    images = listdir(dataset_path)
    date_epoch = []
    logger.info(f'Number of files:{len(images)}')
    for filename in images:
        full_path = path.join(dataset_path, filename)
        try:
            # im = cv.imread(full_path)# try to open the file to test if image
            date = date_splitter(filename)
            if date:
                date_epoch.append(date)
        except Exception as e:
            logger.info(f'Error: {e} on frame:{full_path}')
    
    date_epoch = list(set(date_epoch))
    date_epoch.sort()
    dates = [datetime.utcfromtimestamp(stamp) for stamp in date_epoch]

    logger.info('Grouping observations by date')
    num_obs_date = {} # date: count
    # Hours of observations
    dates_hours = []
    hours = []
    # get months where start and end observations
    init = 12
    end = 0

    # Number of observations by months
    obs_by_month = [0 for i in range(12)]
    for d in dates:
        Y = d.year
        m = d.month
        dd = d.day
        h = d.hour

        if m<init: init = m
        if m>end: end = m
  
        key = f'{Y}-{m}'
        if by == 'day':
            key = f'{m}-{dd}'
        if key in num_obs_date:
            num_obs_date[key] += 1
        else:
            num_obs_date[key] = 1
        
        # For hour distribution on dates
        dates_hours.append(key)
        hours.append(h)

        # for month counting
        obs_by_month[m-1] += 1

    # Add days with 0 observations
    logger.info(f'Compliting observed months: {end-init}')
    months_labels = []
    months_middle = []
    for m in range(init, end+1):
        # get months names and middle for replace axis labels
        months_labels.append(months_names[m-1])
        months_middle.append(f'{m}-15')

        n_days = days[m-1]
        for dd in range(n_days):
            key = f'{m}-{dd+1}'
            if key not in num_obs_date:
                num_obs_date[key] = 0
    # Sort observation count dates
    dates_obs_count = list(num_obs_date.keys())
    dates_obs_count.sort(key=lambda x: (int(x.split('-')[0]),int(x.split('-')[1])))

    obs_count = []
    for k in dates_obs_count:
        obs_count.append(num_obs_date[k])

    # Plotting
    fig = plt.figure(figsize=(30, 9))
    axes = fig.add_subplot(111)

    # Count
    plt.plot(dates_obs_count, obs_count, color=(0,0,0), label='Suma')

    # Fill months total
    prev_date = None
    prev_val = None
    for m in range(init, end+1):
        n_days = days[m-1]
        _dates = [] 
        _count = []
        if prev_date:
            _dates.append(prev_date)
            _count.append(prev_val)
        for dd in range(n_days):
            k = f'{m}-{dd+1}'
            _dates.append(k)
            _count.append(num_obs_date[k])
        plt.fill_between(_dates, _count, )
        prev_date = _dates[-1]
        prev_val = _count[-1]

    # Hours scatter
    plt.scatter(dates_hours, hours, marker='.', label='Hora de observación',  color=(0,0,0))

    # Set months as axis labels
    axes.set_xticks(months_middle)
    axes.set_xticklabels(months_labels, fontsize=16)

    # Time lines
    plt.axhline(y=12, xmin=dates_obs_count[0], xmax=dates_obs_count[-1], linestyle='--', color=(1,0,0.5), label='Medio día')
    plt.axhline(y=min(hours), xmin=dates_obs_count[0], xmax=dates_obs_count[-1], linestyle='--', color=(0.5,0,1), label='Hora min')
    plt.axhline(y=max(hours), xmin=dates_obs_count[0], xmax=dates_obs_count[-1], linestyle='--', color=(0,1,0.5), label='Hora max')

    # plt.ylabel("Número Observaciones", fontsize=18) 
    plt.title("Observaciones por fecha", fontsize=18)

    leg = plt.legend(fontsize=16)
    if save:
        savepath = savepath if savepath else dataset_path
        savefile = path.join(savepath, 'observations-by-date.png')
        logger.info(f'Saving observations at: {savefile}')
        fig.savefig(savefile)
    
    missing_obs_path = path.join(savepath, 'missing-obs.txt')
    logger.info(f'Save missing at {missing_obs_path}')

    with open(missing_obs_path, 'w') as f:
        f.write('Month-Day\n')

        for k,v in num_obs_date.items():
            if v==0:
                f.write(k+'\n')

    # Saving metadata
    month_with_obs = [i for i in range(len(obs_by_month)) if obs_by_month[i] != 0]
    metadata = {
        'number of observations': sum(obs_count),
        'months with observations': month_with_obs,
        'observed months': len(month_with_obs),
        'month average': np.average([obs_by_month[i] for i in month_with_obs]),
        'day average': sum(obs_by_month) / sum([days[i-1] for i in month_with_obs ])
    }
    print(metadata)

    if show: plt.show()