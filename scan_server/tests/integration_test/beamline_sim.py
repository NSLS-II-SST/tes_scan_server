from __future__ import print_function

import socket
import zmq

import signal
from os import path
import os
import sys
from subprocess import call, Popen
import json
import datetime
import yaml
import numpy as np
from time import sleep

offline=True

scan_server = 4000
HTXS_GLOBAL= 1

def formatMsg(method, params):
    msg = {"method": method}
    if params is not None and params != []:
        msg["params"] = params
    return json.dumps(msg).encode()

def send(method, *params):
    msg = formatMsg(method, params)
    s = socket.socket()
    s.connect(("", scan_server))
    s.send(msg)
    s.close()

def sendrcv(method, *params):
    msg = formatMsg(method, params)
    s = socket.socket()
    s.connect(("", scan_server))
    s.send(msg)
    m = s.recv(1024).decode()
    print(m)
    s.close()
    
def take_noise():
    send('start_noise')

def take_filters(name="mixv3", sid=101, htxs_date=None):
    if htxs_date is None:
        htxs_date = datetime.date.today().strftime("%Y%m%d")
    send({"func": "set_htxs", "args": [htxs_date]})
    msg = {"func": "start_cal", "kwargs": {"sample_name": name, "sample_id": sid}}
    send(msg)
    send({"func": "set_cal"})
         
def projectors():
    send({"func": "createProjectors"})

def startSimPulses(amps=[8488, 7050, 6148, 5244, 3922, 2782]):
    send({"func": "publish", "args":["dastard"], "kwargs":{"func":"startSimPulses", "args":[amps]}})

def file_start(path='/tmp'):
    send("file_start", path)

def file_end():
    send("file_end")

def quick_post_process():
    send("quick_post_process")
    
def calibration_start(var_name, var_unit, scan_num, sample_id, sample_name, extra={}, routine='ssrl_10_1_mix_cal'):
    print(f"start calibration scan {scan_num}")
    send("calibration_start", var_name, var_unit, scan_num, sample_id, sample_name, extra, 'none', routine)
    
def scan_start(var_name, var_unit, scan_num, sample_id, sample_name, extra={}):
    print(f"start scan {scan_num}")
    send("scan_start", var_name, var_unit, scan_num, sample_id, sample_name, extra, 'none')

def scan_point_start(var_name, extra={}):
    send("scan_point_start", var_name, extra)

def scan_point_end():
    send("scan_point_end")

def scan_end(try_post_processing=False):
    send("scan_end", try_post_processing)

def calibration_learn_from_last_data():
    send("calibration_learn_from_last_data")

def roi_set(*args):
    send("roi_set", args)

def roi_get_counts():
    return sendrcv("roi_get_counts")

def gscan_grid(gscan_string):
    gscan_args = gscan_string.split()
    gscan_grid = [float(gscan_args[1])]
    pos_init = gscan_grid[0]
    npts_grid = 1
    for n in range(2, len(gscan_args)-2, 2):
        pos_end = float(gscan_args[n])
        stepsize = float(gscan_args[n+1])
        npts_rgn = int((pos_end - pos_init)/stepsize)
        if npts_rgn <= 0:
            continue
        else:
            for ndx in range(npts_rgn):
                gscan_grid.append(gscan_grid[-1] + stepsize)
            npts_grid += npts_rgn
            pos_init = pos_end
    return gscan_grid
    
def runXAS(gscan_args, counts, repeat=2, time=5, qpp=False):
    global HTXS_GLOBAL
    sum_counts = np.sum(counts)
    tot_time = time*60 #minutes to seconds
    sample_name = 'sample'
    sample_id = 101
    var_name = 'mono'
    var_unit = 'eV'

    roi_set((150, 250, "test1"), (350, 450, "test2"))
    mono = gscan_grid(gscan_args)
    for n in range(repeat):
        scan_start(var_name, var_unit, HTXS_GLOBAL, sample_id, sample_name, {'pass': n, 'scantype': 'xas'})
        for m, c in zip(mono, counts):
            dt = c*tot_time/sum_counts
            scan_point_start(m)
            sleep(dt)
            scan_point_end()
            print("ROI Counts", roi_get_counts())
        scan_end()
        if qpp:
            quick_post_process()
        HTXS_GLOBAL += 1

def runXES(npts, dwell=1):
    global HTXS_GLOBAL
    sample_name = 'sample'
    sample_id = 101
    var_name = 'time'
    var_unit = 'eV'
    
    scan_start(var_name, var_unit, HTXS_GLOBAL, sample_id, sample_name, {'scantype': 'xes'})
    for n in range(npts):
        scan_point_start(n)
        sleep(dwell)
        scan_point_end()
    scan_end()
    HTXS_GLOBAL += 1
    
def runCal(npts, dwell=1.0):
    global HTXS_GLOBAL
    sample_name = 'cal_mix'
    sample_id = 101
    var_name = 'time'
    var_unit = 'seconds'
    
    calibration_start(var_name, var_unit, HTXS_GLOBAL, sample_id, sample_name, {}, 'simulated_cal_mix')
    #send({"func": "start_count", "args":["xes"]})
    for n in range(npts):
        scan_point_start(n*dwell)
        sleep(dwell)
        scan_point_end()
    scan_end()
    HTXS_GLOBAL += 1

def testRun(htxs=None):
    global HTXS_GLOBAL
    if htxs is not None:
        HTXS_GLOBAL = htxs
    file_start()
    runCal(30, dwell=1)
    calibration_learn_from_last_data()
    gscan_args = 'mono 700 710 1 720 0.5 1'
    grid = gscan_grid(gscan_args)
    counts = (np.arange(len(grid)) + len(grid))
    runXAS(gscan_args, counts, repeat=2, time=3)
    file_end()

def testXAS(repeat=2, htxs=None):
    global HTXS_GLOBAL
    if htxs is not None:
        HTXS_GLOBAL = htxs
    file_start()
    gscan_args = 'mono 700 710 1 720 0.5 1'
    grid = gscan_grid(gscan_args)
    counts = (np.arange(len(grid)) + len(grid))
    runXAS(gscan_args, counts, repeat=repeat, time=3)
    file_stop()

def testROICounts(htxs=None):
    global HTXS_GLOBAL
    if htxs is not None:
        HTXS_GLOBAL = htxs    
    file_start()
    runCal(5, dwell=1)
    calibration_learn_from_last_data()
    gscan_args = 'mono 700 705 1 1'
    grid = gscan_grid(gscan_args)
    counts = np.ones_like(grid)
    runXAS(gscan_args, counts, repeat=2, time=0.5)
    file_end()

def testQPP(htxs=None):
    # Should really just nuke beamtime_1
    global HTXS_GLOBAL
    if htxs is not None:
        HTXS_GLOBAL = htxs    

    file_start()
    runCal(5, dwell=1)
    calibration_learn_from_last_data()
    gscan_args = 'mono 700 705 1 1'
    grid = gscan_grid(gscan_args)
    counts = np.ones_like(grid)
    runXAS(gscan_args, counts, repeat=2, time=0.5, qpp=True)
    file_end()
    
def testFileStartStop():
    file_start()
    file_end()
    file_start()
    file_end()
