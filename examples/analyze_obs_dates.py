"""Analyze the dates of the observations"""
import logging

from MLanalyzer.auxfunc.tools import analyze_observation_dates


logging.basicConfig(level=logging.DEBUG)

images_path = '/misdoc/datasets/baluarte/from-vrap1'
savepath = '/home/juanc/'

analyze_observation_dates(images_path, by='day', save=True, savepath=savepath, show=False)