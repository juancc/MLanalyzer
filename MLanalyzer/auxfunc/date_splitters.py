"""
Functions for splitting strings into seconds from epoch.
UNIX systems define the epoch as January 1, 1970. The Win32 API, on the other hand, defines the epoch as January 1, 1601.
You can use time.gmtime() to determine your system’s epoch

JCA
Vaico
"""
import time
import datetime


def nvr_default_1(strim, *args, **kwargs):
    """String input: channel4_2020-07-03_14:29:03"""
    date = None
    try:
        _, full_date, full_time = strim.split('_')
        fd = [int(i) for i in full_date.split('-')]
        ft = [int(i) for i in full_time.split(':')]

        d = datetime.datetime(fd[0], fd[1], fd[2], ft[0], ft[1], ft[2])
        date = time.mktime(d.timetuple())
    except Exception as e:
        print(f'Couldnt extract date: {strim}. Err: {e}')
    return date

def vaico_recorder_1(strim, *args, **kwargs):
    """String input: 2018-11-07_15_40_34"""
    date = None
    try:
        full_date, hh, mm, ss = strim.split('_')
        YY,MM,DD = full_date.split('-')

        d = datetime.datetime(int(YY), int(MM), int(DD), int(hh), int(mm), int(ss))
        date = time.mktime(d.timetuple())

    except Exception as e:
        print(f'Couldnt extract date: {strim}. Err: {e}')
    return date




if __name__ == "__main__":
    # d = nvr_default_1('channel4_2020-07-03_14:29:03')
    # print(d)

    d = vaico_recorder_1('2018-11-07_15_40_34')
    print(d)
