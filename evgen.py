#!/usr/bin/python3
import sys
import logging
logger = logging.getLogger(__name__)

from evgen_base import EventGeneratorBase, EventGeneratorApp
from sig_interpolate import InterpSigmaCached


class EventGeneratorFW(EventGeneratorBase):
    """Event generator from helicity amplitudes"""
    def __init__(self, conf):
        super().__init__(conf)
        logger.info("Loading data")
        self.dsigma = InterpSigmaCached('maid', self.channel)
        logger.info("EvGen initialized")

    def get_dsigma(self, event):
        return self.dsigma.interp_dsigma(*event, self.ebeam, h=self.helicity)


if __name__=='__main__':
    EventGeneratorApp.launch(EventGeneratorFW)
