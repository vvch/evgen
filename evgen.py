#!/usr/bin/python3
import logging
logger = logging.getLogger(__name__)
import re
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


class EventGeneratorApp(EventGeneratorApp):
    def get_header(self):
        header = super().get_header()
        a = self.args
        h = a.helicity
        h = f"{h:+}" if h        \
            else 'random +1 or -1'
        header = re.sub(
            r'(Started:.*?\n)',
            f"\1"
            f"Channel:         {a.channel}\n"
            f"Helicity:        {h}\n",
            header
        )
        return header

    def get_arg_parser(self):
        parser = super().get_arg_parser()
        parser.add('--channel', '-C', type=str, required=True,
            choices=InterpSigmaCorrectedCached.available_channels,
            help='Channel')
        parser.add('--helicity', '-H', type=int, default=0,
            choices=[-1, 0, 1],
            help='Electron helicity (use 0 for random choice)')
        return parser


if __name__=='__main__':
    EventGeneratorApp.launch(EventGeneratorFW)
