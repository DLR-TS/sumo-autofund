#!/usr/bin/env python
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

# @file    make-and-sim.py
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
import numpy as np

SUMO_HOME = os.environ.get('SUMO_HOME')
if SUMO_HOME is None:
    sys.exit("please declare environment variable 'SUMO_HOME' or set the path in the script")
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import sumolib

netconvertBinary = sumolib.checkBinary('netconvert')
sumoBinary = sumolib.checkBinary('sumo')

xPos = [0,2000,3000,3000,2000,1000,0,0,1000,2000,3000,3000]
yPos = [0,0,0,1000,1000,1000,1000,2000,2000,2000,2000,1500]

rr = ''
objCnt = 0
cntDump = 0


def get_options(args=None):
    parser = sumolib.options.ArgumentParser(description="Generating single road networks and calculating fundamental diagrams on them")
    parser.add_argument("--optimize", action="store_true", default=False,
                        help="try to find optimal vehicle type parameters to fit HBS")
    parser.add_argument("-l", "--num-lanes", type=int, default=3,
                        help="number of lanes")
    parser.add_argument("--sigma", type=float, default=0.7,
                        help="imperfection parameter sigma")
    parser.add_argument("--speed-deviation", type=float, default=0.1,
                        help="speed deviation parameter")
    parser.add_argument("--step-length", type=float, default=1.,
                        help="simulation step length in s")
    parser.add_argument("--flow-interval", type=float, default=600.,
                        help="time interval for which the flow is constant")
    parser.add_argument("-t", "--tau", type=float, default=1.0,
                        help="aspired minimum time headway in s")
    parser.add_argument("--truck-percentage", type=float, default=0.,
                        help="percentage of trucks")
    parser.add_argument("-s", "--speed-limit", type=float, default=100.,
                        help="speed limit in km/h")
    parser.add_argument("--car-following-model", default="Krauss",
                        help="which car-following model to use")
    parser.add_argument("--basename", help="filename base for outputs")
    parser.add_argument("--mesosim", action="store_true", default=False,
                        help="use mesoscopic simulation")
    parser.add_argument("--vss", action="store_true", default=False,
                        help="use variable speed signs to limit flow")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="tell me what you are doing")

    options = parser.parse_args(args=args)
    if options.basename is None:
        options.basename = "HBS%s" % options.num_lanes
    return options


# helper function
def n2E(name, var):
    result = name + '="' + repr(var) + '" '
    return result

def n2Em(name, var):
    result = name[0] + '="' + repr(var[0]) + '" '
    for i in range(len(name)):
        result = name[i] + '="' + repr(var[i]) + '" '
    return result

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

V0 = [121.20, 121.82, 123.62, 125.26]
L0 = [0.3119, 0.2910, 0.2476, 0.2227]
C0 = [6155, 6101, 5916, 5694]

def vHBS(q, pT):
    indx = int(round(10 * pT))
    if indx > 3:
        indx = 3
    if indx < 0:
        indx = 0
    return V0[indx] / (1 + V0[indx] / (L0[indx] * (C0[indx] - q)))

def readEdgeDataFile(fname):
    vel = []
    flw = []
    for edge in sumolib.xml.parse_fast(fname, "edge", ["density", "speed"]):
        kk = float(edge.density)
        vv = 3.6 * float(edge.speed)
        vel.append(vv)
        flw.append(vv * kk)
    return flw, vel

def compDist(ufName, pT):
    global cntDump

    flw, vel = readEdgeDataFile(ufName)
    nPerBin = 10
    n = len(flw)
    nBin = n // nPerBin
    qmin = 0.9999 * min(flw)
    qmax = 1.0001 * max(flw)
    dx = (qmax - qmin) / nBin
    qA = nBin * [0.0]
    vA = nBin * [0.0]
    cnt = nBin * [0]

    for i in range(len(flw)):
        indx = int((flw[i] - qmin) / dx)
        qA[indx] += flw[i]
        vA[indx] += vel[i]
        cnt[indx] += 1

    fp = open('dump-' + repr(cntDump) + '.txt','w')
    cntDump += 1
    v2sum = 0.0
    cntNonZero = 0
    for i in range(nBin):
        if cnt[i] > 0:
            cntNonZero += 1
            qA[i] /= cnt[i]
            vA[i] /= cnt[i]
            vHat = vHBS(qA[i], pT)
            v2sum += (vA[i] - vHat) * (vA[i] - vHat)
            print(i,qA[i],vA[i],vHat,cnt[i],file=fp)

    if cntNonZero > 0:
        res = math.sqrt(v2sum / cntNonZero)
    else:
        res = 100.0
    print('#error:', res, file=fp)
    fp.close()
    print(res)
    return res


# model can be (element name) 'carFollowing-Krauss', 'carFollowing-BKerner',
# 'carFollowing-IDM'
# (attribut names:) Krauss, BKerner, IDM, Wiedemann, SmartSK, PWagner2009
# lcSpeedGain, lcCooperative, lcKeepRight
def makeCarTypes(model, fBase, fDev, aMax, bMax, sigma, tau, lcSG, pTruck):
    vBase = (fBase[0] - fBase[1] * pTruck) / (1.0 - pTruck)
    spP = n2E('speedFactor',vBase) + n2E('speedDev',fDev[0])
    spT = n2E('speedFactor',fBase[1]) + n2E('speedDev',fDev[1])
    cfModel = 'carFollowModel="' + model + '" '
    pPars = 'minGap="3" ' + spP + n2E('sigma',sigma[0]) + n2E('tau',tau[0]) + n2E('accel',aMax[0]) + n2E('decel',bMax[0])
    lcPars = n2E('lcSpeedGain',lcSG[0])
    tPars = 'minGap="3" ' + spT + n2E('sigma',sigma[1]) + n2E('tau',tau[1]) + n2E('accel',aMax[1]) + n2E('decel',bMax[1])
    pcars = '\t<vType id="pass" maxSpeed="50" length="4.2" ' + cfModel + pPars + lcPars + '/>\n'
    trucks = '\t<vType id="truck" maxSpeed="30" length="11.6" ' + cfModel + tPars + lcPars + '/>'

    return pcars + trucks

def writeVSStimes(vLim, t1, t2):
    factor = [0.5, 0.375, 0.25, 0.18, 0.12, 0.09, 0.06, 0.03, 0, 1]
    dt = (t2 - t1) / len(factor)
    with open('vss-times.xml', 'w') as fvss:
        print('<vss>',file = fvss)
        print('<step ' + n2E('time',0) + n2E('speed',vLim) + '/>', file=fvss)
        for i in range(len(factor)):
            print('<step ' + n2E('time',t1 + i * dt) + n2E('speed',factor[i] * vLim) + '/>', file=fvss)
        print('</vss>',file = fvss)

def makeNet(baseName, nrLanes, speedL=27.7):
    fnod = open(baseName + '.nod.xml','w')
    print('<nodes>',file=fnod)
    for i in range(len(xPos)):
        print('\t<node ' + n2E('id',i) + n2E('x',xPos[i]) + n2E('y',yPos[i]) + '/>', file=fnod)
    print('</nodes>',file=fnod)
    fnod.close()

    rest = n2E('numLanes',nrLanes) + n2E('speed',speedL) + '/>'
    fedg = open(baseName + '.edg.xml','w')
    print('<edges>',file=fedg)
    for i in range(len(xPos) - 1):
        print('\t<edge ' + n2E('from',i) + n2E('id',i) + n2E('to',i + 1) + rest, file=fedg)
    print('</edges>',file=fedg)
    fedg.close()

    print("# preparing the simulation with netconvert...")
    subprocess.call([netconvertBinary, '--junctions.limit-turn-speed', '-1',
                     '-n', baseName + '.nod.xml', '-e', baseName + '.edg.xml', '-o', baseName + '.net.xml'])

def makeDemand(baseName, nrLanes, vLim, pTruck, vss, vtypes, dt):
    # define the one and only route:
    rtxt = '\t<route id="one" edges="'
    for i in range(len(xPos) - 1):
        rtxt += repr(i) + ' '
    rtxt += '"/>'
    rest = ' departPos="last" departSpeed="max"/>'

    with open(baseName + '.rou.xml', 'w') as frou:
        print('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',file=frou)
        if len(vtypes) == 0:
            pcars = '\t<vType id="pass" maxSpeed="50" length="4.5" speedFactor="1.1" />\n'
            trucks = '\t<vType id="truck" maxSpeed="30" length="11.6" />'
            print(pcars + trucks, file=frou)
        else:
            print(vtypes,file=frou)
        print(rtxt, file=frou)
# does work with vehicles, but not with flows (?) would simplify my demand
# computation below
#        print('<vTypeDistribution id="typedist1">',file=frou)
#        print('\t<vType id="pass" ' + n2E('probability',1-pTruck) +
#        '/>',file=frou)
#        print('\t<vType id="truck" ' + n2E('probability',pTruck) +
#        '/>',file=frou)
#        print('</vTypeDistribution>',file=frou)

        cnt = 0
        iFlow = 0
        tmax = 43200
        tmax = 21600
        dFlow = 2400 * dt / tmax
        for t in range(0, tmax, dt):
            iFlow += 1
            flw = iFlow * dFlow
            tflw = pTruck * nrLanes * flw
            pflw = flw - tflw
            dem = ''
            for ll in range(nrLanes):
                if ll == 0:
                    txt = '\t<flow ' + n2E('id',cnt) + ' type="pass" route="one" ' + n2E('begin',t)
                    txt += n2E('end',t + dt) + n2E('departLane',ll) + n2E('vehsPerHour',pflw) + rest
                    dem += txt + '\n'
                    cnt += 1
                    if pTruck > 0:
                        txt = '\t<flow ' + n2E('id',cnt) + ' type="truck" route="one" ' + n2E('begin',t)
                        txt += n2E('end',t + dt) + n2E('departLane',ll) + n2E('vehsPerHour',tflw) + rest
                        dem += txt + '\n'
                        cnt += 1
                else:
                    txt = '\t<flow ' + n2E('id',cnt) + ' type="pass" route="one" ' + n2E('begin',t)
                    txt += n2E('end',t + dt) + n2E('departLane',ll) + n2E('vehsPerHour',flw) + rest
                    dem += txt + '\n'
                    cnt += 1

            print(dem, file=frou)

## keep flw at maximum
        t1 = tmax
        t2 = tmax + 10800
        if vss > 0:
            dem = ''
            for ll in range(nrLanes):
                if ll == 0:
                    txt = '\t<flow ' + n2E('id',cnt) + ' type="pass" route="one" ' + n2E('begin',t1)
                    txt += n2E('end',t2) + n2E('departLane',ll) + n2E('vehsPerHour',pflw) + rest
                    dem += txt + '\n'
                    cnt += 1
                    if pTruck > 0:
                        cnt += 1
                        txt = '\t<flow ' + n2E('id',cnt) + ' type="truck" route="one" ' + n2E('begin',t1)
                        txt += n2E('end',t2) + n2E('departLane',ll) + n2E('vehsPerHour',tflw) + rest
                        dem += txt + '\n'
                        cnt += 1
                else:
                    txt = '\t<flow ' + n2E('id',cnt) + ' type="pass" route="one" ' + n2E('begin',t1)
                    txt += n2E('end',t2) + n2E('departLane',ll) + n2E('vehsPerHour',flw) + rest
                    dem += txt + '\n'
                    cnt += 1
            print(dem, file = frou)

        print('</routes>',file=frou)
        frou.close()

        if vss > 0:
            writeVSStimes(vLim, t1, t2)



def makeAdditionalFiles(baseName, extName, nrLanes, vss, loops, dt):
    n = len(xPos) - 2
    if loops > 0:
        with open(baseName + '.loops.xml','w') as floop:
            print('<additional>', file = floop)
            rest = 'pos="500.00" freq="3600" friendly_pos="x" splitByType="x" file="detA' + extName + '.out.xml"/>'
            for i in range(1,n):
                for l in range(nrLanes):
                    il = repr(i) + '_' + repr(l)
                    print('\t<inductionLoop id="%s" lane="%s" ' % (il,il) + rest, file = floop)
                print('\t<e3Detector id="e3_%s" freq="%s" file="detE3%s.out.xml">' % (i, dt, extName), file = floop)
                for l in range(nrLanes):
                    print('\t\t<detEntry pos="250" lane="%s_%s"/>' % (i,l), file = floop)
                    print('\t\t<detExit pos="750" lane="%s_%s"/>' % (i,l) + rest, file = floop)
                print('\t</e3Detector>', file = floop)
            print('</additional>', file = floop)

    with open(baseName + '.add.xml','w') as fadd:
        print('<additional>',file=fadd)
        if vss > 0:
            for l in range(nrLanes):
                txt = 'lanes="' + repr(n) + '_' + repr(l) + '" file="vss-times.xml"/>'
                print('\t<variableSpeedSign id="' + repr(l) + '" ' + txt,file=fadd)

        print('\t<edgeData id="ed" ' + n2E('freq',dt) + ' file="edgeStats' + extName + '.xml" excludeEmpty="true"/>',  file=fadd)
        print('\t<laneData id="ld" ' + n2E('freq',dt) + ' file="laneStats' + extName + '.xml" excludeEmpty="true"/>',  file=fadd)
        print('</additional>',file=fadd)


def makeConfig(baseName, deltaT, loops):
    addF = baseName + '.add.xml'
    if loops > 0:
        addF += ',' + baseName + '.loops.xml'

    open(baseName + '.sumocfg','w').write("""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="%s.net.xml"/>
        <route-files value="%s.rou.xml"/>
        <additional-files value="%s"/>
    </input>
    <time>
        <step-length value="%s"/>
    </time>
    <processing>
        <max-depart-delay value="10"/>
    </processing>
    <report>
        <xml-validation value="never"/>
        <duration-log.disable value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>
""" % (baseName, baseName, addF, deltaT))

def runIt(baseName, myExt, nLanes, speedLimit, pTruck, dT, vTypes, meso, vss):
    loops = 1
    makeNet(baseName, nLanes, speedLimit)
    dt = 600
    makeDemand(baseName, nLanes, speedLimit, pTruck, vss, vTypes, dt)

    makeAdditionalFiles(baseName, myExt, nLanes, vss, loops, dt)
    makeConfig(baseName, dT, loops)
    xml2csv = os.path.join(SUMO_HOME, 'tools', 'xml', 'xml2csv.py')

    if meso > 0:
        subprocess.call([sumoBinary, '-c', baseName + '.sumocfg', '--mesosim', '--meso-overtaking'])
        subprocess.call(['python', xml2csv, 'edgeStats' + myExt + '.xml'])
        print("meso done")
    else:
        subprocess.call([sumoBinary, '-c', baseName + '.sumocfg'])
#  useful for debugging -- just a short run...
#	retcode = subprocess.call([sumoBinary, '-c', baseName + '.sumocfg', '--end', str(3600)])
        subprocess.call(['python', xml2csv, 'edgeStats' + myExt + '.xml'])
        subprocess.call(['python', xml2csv, 'laneStats' + myExt + '.xml'])
        print("micro done")

## The following, so far, runs only on my linux box at home
def saltelliSA(nSim, pTruck, speedLimit):
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    basename = 'sensitivity'
    nLanes = 3
    dT = 1
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

    for i in range(len(Y)):
        myExt = '-' + repr(i) + '-micro'
        myExtMeso = '-' + repr(i) + '-meso'

        rr = makeCarTypes('Krauss', [X[i,6],X[i,7]], [X[i,8],X[i,9]], [X[i,0],X[i,1]], [X[i,2],X[i,3]], [X[i,10],X[i,11]], [X[i,4],X[i,5]], [X[i,12],X[i,13]], pTruck)
        runIt(basename, myExt, nLanes, speedLimit, pTruck, dT, rr, 0, 0)
        err = compDist('edgeStats' + myExt + '.xml', pTruck)

        runIt(basename, myExtMeso, nLanes, speedLimit, pTruck, dT, rr, 1, 0)
        errM = compDist('edgeStats' + myExtMeso + '.xml', pTruck)

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

def simpleSensitivity(nSim, pTruck, speedLimit):
    basename = 'sensitivity'
    flog = open(basename + '.log','w')

    nLanes = 3
    x = halton(14, nSim)
    for i in range(nSim):
        a = [1 + 2 * x[i,0], 0.5 + x[i,1]]
        b = [2 + 4 * x[i,2], 1 + 5 * x[i,3]]
        ttau = [1 + x[i,4], 1 + x[i,5]]
        fBase = [1 + 0.3 * x[i,6], 0.8 + 0.2 * x[i,7]]
        fDev = [0.3 * x[i,8], 0.2 * x[i,9]]
        sig = [x[i,10], x[i,11]]
        lcSG = [1 + 0.4 * x[i,12], 1 + 0.4 * x[i,13]]
        myExt = '-' + repr(i) + '-micro'
        myExtMeso = '-' + repr(i) + '-meso'

        rr = makeCarTypes('Krauss', fBase, fDev, a, b, sig, ttau, lcSG, pTruck)
        runIt(basename, myExt, nLanes, speedLimit, pTruck, dT, rr, 0, 0)
        err = compDist('edgeStats' + myExt + '.xml', pTruck)

        runIt(basename, myExtMeso, nLanes, speedLimit, pTruck, dT, rr, 1, 0)
        errM = compDist('edgeStats' + myExtMeso + '.xml', pTruck)

        print(i, err, a[0], a[1], b[0], b[1], ttau[0], ttau[1], fBase[0], fBase[1], fDev[0], fDev[1], sig[0], sig[1], lcSG[0], lcSG[1], file = flog)
        print(i, errM, a[0], a[1], b[0], b[1], ttau[0], ttau[1], fBase[0], fBase[1], fDev[0], fDev[1], sig[0], sig[1], lcSG[0], lcSG[1], file = flog)
        flog.flush()

    flog.close()

def myBase():
    nLanes = 3
    sigma = 0.91
    dT = 1
    vDev = 0.12
    basename = 'HBS' + repr(nLanes)
    speedLimit = 100 / 3.6
    vss = 0

    a = [2, 1]
    b = [4.5, 2.5]
    ttau = [1.0, 1.0]
    fBase = [1.19, 0.89]
    fDev = [vDev, 0.06]
    sig = [sigma, 0.73]
    lcSG = [1.17, 1.1]

    for iTruck in range(4):
        pTruck = 0.1 * iTruck
        print(pTruck, vDev, sigma)
        rr = makeCarTypes('Krauss', fBase, fDev, a, b, sig, ttau, lcSG, pTruck)

        meso = 0
        ext = '-' + str("%.2f" % sigma) + '_' + str("%.2f" % pTruck) + '_' + str("%.2f" % vDev)
        runIt(basename, ext, nLanes, speedLimit, pTruck, dT, rr, meso, vss)

        meso = 1
        ext = '-meso-' + str("%.2f" % pTruck) + '_' + str("%.2f" % vDev)
        runIt(basename, ext, nLanes, speedLimit, pTruck, dT, rr, meso, vss)


def sumoDefault(pTruck=0.0, vss=0, speedDev=-1):
    rr = ''
    nLanes = 3
    dT = 1
    speedLimit = 100 / 3.6
    basename = 'HBSdef' + repr(nLanes)
    meso = 0
    ext = '-default'
    runIt(basename, ext, nLanes, speedLimit, pTruck, dT, rr, meso, vss)

    meso = 1
    ext = '-default-meso'
    runIt(basename, ext, nLanes, speedLimit, pTruck, dT, rr, meso, vss)

def platooning():
    nLanes = 3
    sigma = 0.5
    dT = 1
    vDev = 0.24
    basename = 'HBS' + repr(nLanes)
    speedLimit = 130 / 3.6
    vss = 0

    a = [2.6, 2.6]
    b = [3.5, 3.5]
    ttau = [1.0, 1.0]
    fBase = [0.8,0.65]
    fDev = [vDev, 0.2]
    sig = [sigma, sigma]
    lcSG = [1, 1]

    pTruck = 0.2
    rr = makeCarTypes('Krauss', fBase, fDev, a, b, sig, ttau, lcSG, pTruck)

    meso = 0
    ext = '-pulk'
    runIt(basename, ext, nLanes, speedLimit, pTruck, dT, rr, meso, vss)

    meso = 1
    ext = '-pulk-meso'
    runIt(basename, ext, nLanes, speedLimit, pTruck, dT, rr, meso, vss)


def objectNL(x, grad):
    global objCnt
    objCnt += 1
    basename = 'HBSopt'
    ext = '-optim'
    nLanes = 3
    pTruck = 0.1
    dT = 1.0
    speedLimit = 100 / 3.6
    a = [x[0], x[1]]
    b = [x[2], x[3]]
    ttau = [x[4], x[5]]
    fDev = [x[6],x[7]]
    fBase = [x[8], x[9]]
    sig = [x[10], x[11]]
    lcSG = [x[12], x[13]]
    rr = makeCarTypes('Krauss', fBase, fDev, a, b, sig, ttau, lcSG, pTruck)

    runIt(basename, ext, nLanes, speedLimit, pTruck, dT, rr, 0, 0)
    err = compDist('edgeStats' + ext + '.xml', pTruck)

    with open('optimization.log','a') as flog:
        res = repr(objCnt) + ' ' + repr(err) + ' '
        for i in range(len(x)):
            res += repr(x[i]) + ' '
        print(res, file = flog)
    flog.close()
    print(res)

    return err

def f1Dev(x, grad):
    return x[0]

def f2Dev(x):
    return x[1]

def lc1Cstrt(x):
    return x[2] - 1

def lc2Cstrt(x):
    return x[3] - 1

def optimization():
    import nlopt
    with open('optimization.log','w') as flog:
        print('# parameters objective')
    flog.close()
    opt = nlopt.opt(nlopt.LN_BOBYQA,14)
    xlow = [0.5,0.5, 1, 1, 1, 1, 0.01, 0.01, 1.1, 0.8, 0, 0, 0.8, 0.8]
    xhigh = [4.5,4.5, 9, 7, 2, 2, 0.4, 0.2, 1.4, 1, 1, 1, 2, 2]
    opt.set_lower_bounds(xlow)
    opt.set_upper_bounds(xhigh)
    opt.set_min_objective(objectNL)
    opt.set_xtol_rel(1e-4)
    x0 = [2.5,1.5, 4.5, 2.5, 1.0, 1.0, 0.14, 0.06, 1.2, 0.9, 0.9, 0.7, 1.1, 1.1]

    x = opt.optimize(x0)
    minf = opt.last_optimum_value()

    res = 'optimum at: '
    for j in range(len(x0)):
        res += repr(x[j]) + ' '
    print(res)

    print("minimum value = ", minf)
    print("result code = ", opt.last_optimize_result())

def objectiveSC(x):
    grad = 0
    e = objectNL(x, grad)
    return e

def optimizationSC():
    with open('optimization.log','w') as flog:
        print('# parameters objective')
    flog.close()

    from scipy.optimize import fmin_cobyla
    fmin_cobyla(objective, [0.18,0.09,1.1,1.1], [f1Dev, f2Dev, lc1Cstrt, lc2Cstrt], rhobeg=0.2, rhoend=1e-4)


def main(options):
    #optimization()
    #exit(1)

    #platooning()
    #exit(1)

    #sumoDefault()
    #exit(1)

    #myBase()
    #exit(1)

    #saltelliSA(50, 0.1, 100/3.6)
    #exit(1)

    #sensitivity(1000, 0.1)
    #exit(1)

    a = [2.0, 1.0]
    b = [4.5, 3.5]
    ttau = [options.tau, 1.25]
    fBase = [1.18, 0.9]
    fDev = [options.speed_deviation, 0.1]
    sig = [options.sigma, 0.7]
    lcSG = [1, 1]
    rr = makeCarTypes('Krauss', fBase, fDev, a, b, sig, ttau, lcSG, options.truck_percentage)

    ext = '-%.2f_%.2f_%.2f' % (options.sigma, options.truck_percentage, fDev[0])
    runIt(options.basename, ext, options.num_lanes, options.speed_limit / 3.6,
          options.truck_percentage, options.step_length, rr, options.mesosim, options.vss)
    compDist('edgeStats' + ext + '.xml', options.truck_percentage)
    if options.verbose:
        print("ALL DONE")


if __name__ == "__main__":
    main(get_options())
