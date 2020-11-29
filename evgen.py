#!/usr/bin/python3
import sys
import logging
logger = logging.getLogger(__name__)

from evgen_base import EventGeneratorBase, EventGeneratorApp

sys.path.append('/media/vitaly/work/work/jlab/clasfw')

from sig_interpolate import InterpSigmaLinearND


class EventGeneratorFW(EventGeneratorBase):
    """Event generator from helicity amplitudes"""
    def __init__(self, conf):
        super().__init__(conf)
        logger.info("Loading data")
        from clasfw.models import Amplitude, Model, Channel
        from clasfw.app import create_app
        app = create_app()
        with app.test_request_context():
            self.dsigma = InterpSigmaLinearND(
                Amplitude,
                Model.by_name('maid'),
                Channel.by_name(self.channel))
        logger.info("EvGen initialized")

    def get_dsigma(self, event):
        return self.dsigma.interp_dsigma(*event, self.ebeam, h=1)


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
