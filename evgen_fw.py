#!/usr/bin/python3
import sys
import numpy as np
import logging
logger = logging.getLogger(__name__)

from evgen import EventGeneratorBase

sys.path.append('/media/vitaly/work/work/jlab/clasfw')

import hep
from sig_interpolate import InterpSigma, InterpSigmaLinearND


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
            #self.dsigma = InterpSigma(Amplitude, model, channel)
            self.dsigma = InterpSigmaLinearND(Amplitude, model, channel)

    def get_max_dsigma(self):
        # TODO!
       return self.dsigma_max

    def get_dsigma(self, W, Q2, cos_theta, phi):
        Eb = self.ebeam
        h = 1
        eps_T = hep.ε_T(W, Q2, Eb)
        return self.dsigma.interp_dsigma(Q2, W, cos_theta, phi, eps_T, h)


if __name__=='__main__':

    import yaml
    from pathlib import Path

    logging.basicConfig(level=logging.DEBUG)
    #logging.basicConfig(level=logging.INFO)

    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    import coloredlogs
    coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s')

    import argparse

    parser = argparse.ArgumentParser(
        description='Event generator from helicity amplitudes.')
    parser.add_argument('--events', '-n', type=int,
        help='Number of events to generate')
    parser.add_argument('--ebeam', '-E', type=float,
        help='Beam energy, GeV')
    parser.add_argument('--q2min', type=float,
        help='Q^2 min, GeV^2')
    parser.add_argument('--q2max', type=float,
        help='Q^2 max, GeV^2')
    parser.add_argument('--wmin', type=float,
        help='W min, GeV')
    parser.add_argument('--wmax', type=float,
        help='W max, GeV')
    parser.add_argument('--channel', type=str,
        help='Channel')
    parser.add_argument('--output', '-o', type=str,
        default='wq2.dat',
        help='Output file name')
    args = parser.parse_args()

    from estimate_time import EstimateTime
    #from hist_root import Hists4
    logger.debug("Modules loaded")


    with open('evgen.yaml') as f:
        conf = yaml.load(f)
    for attr in 'events ebeam wmin wmax q2min q2max'.split():
        if hasattr(args, attr) and getattr(args, attr, None) is not None:
            conf[attr] = getattr(args, attr)
    logger.info(conf)

    EvGen = EventGeneratorFW(conf)
    timer = EstimateTime(EvGen.events)
    timer.min_interval_to_output = 5  #  sec
    #hist = Hists4()

    evs = []
    for ev in EvGen.generate_events():
        evs.append(ev)
        W, Q2, cos_theta, phi = ev
        #hist.Fill(W, Q2)

        timer.update()
        if timer.may_output():
            print("{:3.0f}%\tCounter: {}\tElapsed: {:8}\t Estimated: {:8}\tPer 1000 events: {:.3g}".format(
                timer.percent, timer.counter,
                timer.elapsed, timer.estimated,
                (timer.elapsed_s / timer.counter)*1000
            ))

    print("Generated: {} events, time: {}".format(
        len(evs), timer.elapsed))
    #hist.save()
    np.savetxt(args.output, evs)
    logger.debug("Done")
