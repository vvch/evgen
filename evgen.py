#!/usr/bin/python3
import sys
from evgen_base import EventGeneratorBase, EventGeneratorApp

sys.path.append('/media/vitaly/work/work/jlab/clasfw')

import hep
from sig_interpolate import InterpSigmaLinearND
import logging
logger = logging.getLogger(__name__)


class EventGeneratorFW(EventGeneratorBase):
    description = 'Event generator from MAID helicity amplitudes'
    def __init__(self, conf):
        super().__init__(conf)
        logger.info("Loading data")
        self.load_data()
        logger.info("EvGen initialized")
        logger.info(f"dsigma_upper = {self.dsigma_upper}")
        logger.debug(f"events = {self.events}")

    def load_data(self):
        from clasfw.models import Amplitude, Model, Channel
        from clasfw.app import create_app
        app = create_app()
        with app.test_request_context():
            model = Model.by_name('maid')
            channel = Channel.by_name(self.channel)
            #self.dsigma = InterpSigma(Amplitude, model, channel)
            self.dsigma = InterpSigmaLinearND(Amplitude, model, channel)

    def get_dsigma(self, event):
        W, Q2, cos_theta, phi = event
        return self.dsigma.interp_dsigma(
            W, Q2, cos_theta, phi,
            hep.ε_T(W, Q2, self.ebeam), h=1)


if __name__=='__main__':
    try:
        EventGeneratorApp(
            EventGeneratorFW,
            log_level=logging.INFO
            #log_level=logging.DEBUG
        ).run()
    except (NotImplementedError, ModuleNotFoundError) as e:
        logger.fatal(e)
        sys.exit(1)
