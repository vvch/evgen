#!/usr/bin/python3
import logging
logger = logging.getLogger(__name__)
import numpy as np
from evgen_base import EventGeneratorBase, EventGeneratorApp
from sig_interpolate import InterpSigmaCorrectedCached


class EventGeneratorFW(EventGeneratorBase):
    """Event generator from helicity amplitudes"""
    def __init__(self, conf):
        super().__init__(conf)
        logger.info("Loading data")
        self.dsigma = InterpSigmaCorrectedCached('maid', self.channel)
        logger.info("EvGen initialized")

    def get_dsigma(self, event):
        h = self.helicity
        if h==0:
            h = np.random.choice((-1, 1))
        return self.dsigma.interp_dsigma_e(*event, self.ebeam, h)


if __name__=='__main__':
    EventGeneratorApp.launch(EventGeneratorFW)
