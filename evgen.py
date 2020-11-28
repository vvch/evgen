#!/usr/bin/python3
import sys
from evgen_base import EventGeneratorBase, EventGeneratorApp

sys.path.append('/media/vitaly/work/work/jlab/clasfw')

import hep
from sig_interpolate import InterpSigmaLinearND
import logging
logger = logging.getLogger(__name__)


class EventGeneratorMAID(EventGeneratorBase):
    description = 'Event generator from MAID helicity amplitudes'
    def __init__(self, conf):
        super().__init__(conf)
        logger.info("Loading data")
        self.load_data()
        logger.info("EvGen initialized")
        logger.info(f"dsigma_max = {self.dsigma_max}")
        logger.debug(f"events = {self.events}")

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

    def get_dsigma(self, W, Q2, cos_theta, phi):
        return self.dsigma.interp_dsigma(
            Q2, W, cos_theta, phi,
            hep.ε_T(W, Q2, self.ebeam), h=1)


if __name__=='__main__':
    try:
        EventGeneratorApp(
            EventGeneratorMAID,
            log_level=logging.INFO
            #log_level=logging.DEBUG
        ).run()
    except (NotImplementedError, ModuleNotFoundError) as e:
        logger.fatal(e)
        sys.exit(1)
