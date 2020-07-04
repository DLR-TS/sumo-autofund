#!/usr/bin/env python3
# Copyright (C) 2019-2020 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

# @file    sensitivity.py
# @author  Peter Wagner
# @author  Michael Behrisch
# @date    2019
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import subprocess
import math
import csv
import multiprocessing
import numpy as np

import make_sim


# the halton code is from here:
# https://mail.scipy.org/pipermail/scipy-user/2013-June/034744.html
# I have extended the list of prime number P allowing for a larger dimension
# than just 12
def halton(dim, nbpts):
    h = np.empty(nbpts * dim)
    h.fill(np.nan)
    p = np.empty(nbpts)
    p.fill(np.nan)
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73]
    lognbpts = math.log(nbpts + 1)
    for i in range(dim):
        b = P[i]
        n = int(math.ceil(lognbpts / math.log(b)))
        for t in range(n):
            p[t] = math.pow(b, -(t + 1))

        for j in range(nbpts):
            d = j + 1
            sum_ = math.fmod(d, b) * p[0]
            for t in range(1, n):
                d = math.floor(d / b)
                sum_ += math.fmod(d, b) * p[t]

            h[j * dim + i] = sum_

    return h.reshape(nbpts, dim)


def saltelliSA(options):
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    nSim = options.num_sims if options.num_sims else 1000
    pTruck = options.truck_percentage
    speedLimit = options.speed_limit / 3.6
    nLanes = options.num_lanes
    basename = 'sensitivity'
    flog = open(basename + '.log','w')

    problem = {
        'num_vars': 14,
        'names': ['a1', 'a2', 'b1', 'b2', 'tau1', 'tau2', 'f1', 'f2','df1', 'df2','sig1','sig2','lc1SG','lc2SG'],
        'bounds': [[1, 4],[0.5,2],[1,9],[1,7],[1,2],[1,2],[1,1.5],[0.8,1],[0.05,5],[0.05,1.5],[0,1],[0,1],[0.8,2],[0.8,2]]
    }

    X = saltelli.sample(problem, nSim, calc_second_order=True)

# Run the "model" -- this will happen offline for external models
# Y = X[:, 0] + (0.1 * X[:, 1]) + ((1.2 * X[:, 2]) * (0.2 + X[:, 0]))

# Run model (example)
#    Y = Ishigami.evaluate(param_values)
    Y = X[:,0]
    Ym = X[:,0]

    make_sim.makeNet(basename, nLanes, speedLimit)
    for i in range(len(Y)):
        myExt = '-' + repr(i) + '-micro'
        myExtMeso = '-' + repr(i) + '-meso'

        rr = make_sim.makeCarTypes('Krauss', [X[i,6],X[i,7]], [X[i,8],X[i,9]], [X[i,0],X[i,1]], [X[i,2],X[i,3]], [X[i,10],X[i,11]], [X[i,4],X[i,5]], [X[i,12],X[i,13]], pTruck, pAuto=0)
        edgeStatsFile = make_sim.runIt(basename, myExt, nLanes, speedLimit, pTruck, vTypes=rr)
        err = make_sim.compDist(edgeStatsFile, pTruck)

        edgeStatsFile = make_sim.runIt(basename, myExtMeso, nLanes, speedLimit, pTruck, vTypes=rr, meso=True)
        errM = make_sim.compDist(edgeStatsFile, pTruck)

        txt = repr(i) + ' ' + repr(err) + ' ' + repr(errM) + ' '
        for j in range(len(X[i,])):
            txt += repr(X[i,j]) + ' '
        print(txt, file = flog)
        flog.flush()
        Y[i] = err
        Ym[i] = errM

# Perform analysis
    uSi = sobol.analyze(problem, X, Y, print_to_console=True)
    mSi = sobol.analyze(problem, X, Ym, print_to_console=True)
    ## Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
    ## (first and total-order indices with bootstrap confidence intervals)


def simpleSensitivity(options):
    nSim = options.num_sims if options.num_sims else 1000
    pTruck = options.truck_percentage
    speedLimit = options.speed_limit / 3.6
    nLanes = options.num_lanes
    basename = 'sensitivity'
    flog = open(basename + '.log','w')

    x = halton(14, nSim)
    make_sim.makeNet(basename, nLanes, speedLimit)
    for i in range(nSim):
        a = [1 + 2 * x[i,0], 0.5 + x[i,1]]
        b = [2 + 4 * x[i,2], 1 + 5 * x[i,3]]
        ttau = [1 + x[i,4], 1 + x[i,5], 1.0]
        fBase = [1 + 0.3 * x[i,6], 0.8 + 0.2 * x[i,7]]
        fDev = [0.3 * x[i,8], 0.2 * x[i,9]]
        sig = [x[i,10], x[i,11]]
        lcSG = [1 + 0.4 * x[i,12], 1 + 0.4 * x[i,13]]
        myExt = '-' + repr(i) + '-micro'
        myExtMeso = '-' + repr(i) + '-meso'

        rr = make_sim.makeCarTypes('Krauss', fBase, fDev, a, b, sig, ttau, lcSG, pTruck, pAuto=0)
        edgeStatsFile = make_sim.runIt(basename, myExt, nLanes, speedLimit, pTruck, vTypes=rr)
        err = make_sim.compDist(edgeStatsFile, pTruck)

        edgeStatsFile = make_sim.runIt(basename, myExtMeso, nLanes, speedLimit, pTruck, vTypes=rr, meso=True)
        errM = make_sim.compDist(edgeStatsFile, pTruck)

        print(i, err, a[0], a[1], b[0], b[1], ttau[0], ttau[1], fBase[0], fBase[1], fDev[0], fDev[1], sig[0], sig[1], lcSG[0], lcSG[1], file = flog)
        print(i, errM, a[0], a[1], b[0], b[1], ttau[0], ttau[1], fBase[0], fBase[1], fDev[0], fDev[1], sig[0], sig[1], lcSG[0], lcSG[1], file = flog)
        flog.flush()

    flog.close()


def main(options):
    if options.saltelli:
        saltelliSA(options)
    else:
        simpleSensitivity(options)


if __name__ == "__main__":
    main(make_sim.get_options())
