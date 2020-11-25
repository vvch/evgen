#!/usr/bin/python3
import sys
import numpy as np
import logging
logger = logging

from evgen import EventGeneratorBase

sys.path.append('/media/vitaly/work/work/jlab/clasfw')

import hep
from sig_interpolate import InterpSigma


class EventGeneratorFW(EventGeneratorBase):
    def __init__(self, conf):
        super().__init__(conf)
        self.load_data()
        self.dsigma_max = 10  #  µb/sr
        logger.info("EvGen initialized.")
        logger.info(f"dsigma_max = {self.dsigma_max}")
        logger.info(f"events = {self.events}")

    def load_data(self):
        from clasfw.models import Amplitude, Model, Channel
        from clasfw.app import create_app
        app = create_app()
        with app.test_request_context():
            model = Model.query.filter_by(name='maid').one()
            channel = Channel.query.filter_by(
                name=self.channel
            ).one()
            self.dsigma = InterpSigma(Amplitude, model, channel)

    def get_max_dsigma(self):
        # TODO!
       return self.dsigma_max

    def get_dsigma(self, W, Q2, cos_theta, phi):
        #Eb = 10.6  #  GeV
        Eb = self.ebeam
        h = 1
        eps_T = hep.ε_T(W, Q2, Eb)
        return self.dsigma.interp_dsigma(Q2, W, cos_theta, phi, eps_T, h)


if __name__=='__main__':

    import yaml
    from pathlib import Path

    logger.basicConfig(level=logging.DEBUG)
    #logger.basicConfig(level=logging.INFO)
    import coloredlogs
    coloredlogs.install()

    import argparse

    parser = argparse.ArgumentParser(
        description='Event generator from helicity amplitudes.')
    parser.add_argument('--events', '-n', type=int,
        help='Number of events to generate')
    parser.add_argument('--ebeam', '-E', type=float,
        help='Beam energy')
    parser.add_argument('--q2min', type=float,
        help='Q^2 min')
    parser.add_argument('--q2max', type=float,
        help='Q^2 max')
    args = parser.parse_args()
    #print(args)

    from estimate_time import EstimateTime
    #from hist_root import Hists4
    logger.info("Modules loaded")


    with open('evgen.yaml') as f:
        conf = yaml.load(f)
    for attr in 'events ebeam wmin wmax q2min q2max'.split():
        if hasattr(args, attr) and getattr(args, attr, None) is not None:
            conf[attr] = getattr(args, attr)
    logger.info(conf)

    EvGen = EventGeneratorFW(conf)
    #EvGen.events = 300
    timer = EstimateTime(EvGen.events)
    #hist = Hists4()

    evs = []
    for ev in EvGen.generate_events():
        evs.append(ev)
        W, Q2, cos_theta, phi = ev
        #hist.Fill(W, Q2)

        timer.update()
        print("Counter: {}\tElapsed: {:8}\t Estimated: {:8}\tPer event: {}".format(
            timer.counter,
            timer.elapsed,
            timer.estimated,
            timer.elapsed_s / timer.counter))

    print("Generated: {} events, time: {}".format(
        len(evs), timer.elapsed))
    #hist.save()
    np.savetxt('wq2.dat', evs)
