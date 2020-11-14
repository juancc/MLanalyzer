from os import path, environ, listdir, makedirs
from datetime import datetime

# import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn')

from MLanalyzer.auxfunc.date_splitters import nvr_default_1

def analyze_observation_dates(dataset_path, date_splitter=nvr_default_1, by='day', save=False):
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

    print('Saving observations by date')
    num_obs_date = {}
    for d in dates:
        Y = d.year
        m = d.month
        d = d.day
        key = f'{Y}-{m}'
        if by == 'day':
            key = f'{m}-{d}'
        if key in num_obs_date:
            num_obs_date[key] += 1
        else:
            num_obs_date[key] = 1
    
    fig = plt.figure(figsize=(20, 6))
    plt.plot(num_obs_date.keys(), num_obs_date.values())
    plt.ylabel("Número Observaciones") 
    plt.title("Observaciones por fecha")
    if save:
        fig.savefig(path.join(dataset_path, 'observations-by-date.png'))
    plt.show()