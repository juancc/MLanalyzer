"""Analyze the dates of the observations"""

from MLanalyzer.auxfunc.tools import analyze_observation_dates

images_path = '/misdoc/datasets/baluarte/analysis-test-3'

analyze_observation_dates(images_path, by='day', save=False)