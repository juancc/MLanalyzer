from os import path, environ, listdir, makedirs
from datetime import datetime

# import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn')

from MLanalyzer.auxfunc.date_splitters import nvr_default_1

def analyze_observation_dates(dataset_path, date_splitter=nvr_default_1, by='day', save=False, savepath=None, show=True):
    """Plot the distribution of the images dates"""
    print(f'Analyzing dates on: {dataset_path}')
    images = listdir(dataset_path)
    date_epoch = []
    print(f'Number of files:{len(images)}')
    for filename in images:
        full_path = path.join(dataset_path, filename)
        try:
            # im = cv.imread(full_path)# try to open the file to test if image
            date = date_splitter(filename)
            if date:
                date_epoch.append(date)
        except Exception as e:
            print(f'Error: {e} on frame:{full_path}')
    
    date_epoch = list(set(date_epoch))
    date_epoch.sort()
    dates = [datetime.fromtimestamp(stamp) for stamp in date_epoch]

    print('Grouping observations by date')
    num_obs_date = {} # date: count
    # Hours of observations
    dates_hours = []
    hours = []
    for d in dates:
        Y = d.year
        m = d.month
        dd = d.day
        h = d.hour
  
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
    
    fig = plt.figure(figsize=(20, 6))

    # Count
    plt.plot(num_obs_date.keys(), num_obs_date.values(), color=(0,0,0), label='Suma')
    # Hours scatter
    plt.scatter(dates_hours, hours, marker='.', label='Hora de observación',  color=(1,0,0))

    # Time lines
    plt.axhline(y=12, xmin=dates_hours[0], xmax=dates_hours[-1], linestyle='--', color=(1,0,0.5), label='Medio día')
    plt.axhline(y=min(hours), xmin=dates_hours[0], xmax=dates_hours[-1], linestyle='--', color=(0.5,0,1), label='Hora min')
    plt.axhline(y=max(hours), xmin=dates_hours[0], xmax=dates_hours[-1], linestyle='--', color=(0,1,0.5), label='Hora max')

    plt.ylabel("Número Observaciones") 
    plt.title("Observaciones por fecha")

    leg = plt.legend(loc=2)
    if save:
        savepath = savepath if savepath else dataset_path
        savefile = path.join(savepath, 'observations-by-date.png')
        print(f'Saving observations at: {savefile}')
        fig.savefig(savefile)
    
    if show: plt.show()