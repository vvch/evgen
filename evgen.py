#!/usr/bin/python3
import sys
from evgen_base import EventGeneratorBase, EventGeneratorApp

sys.path.append('/media/vitaly/work/work/jlab/clasfw')

import hep
from sig_interpolate import InterpSigmaLinearND
import logging
logger = logging.getLogger(__name__)


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
    EventGeneratorApp(EventGeneratorFW, log_level=logging.DEBUG).run()
