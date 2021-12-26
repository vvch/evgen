#!/usr/bin/python3
import numpy as np
import logging
logger = logging.getLogger(__name__)

from evgen import EventGeneratorApp
from hist import Hists4


class EventGeneratorApp(EventGeneratorApp):
    def run(self):
        self.hist = hist = Hists4()
        hist.CreateCanvas()
        hist.Draw()
        super().run()
        hist.save('events_hist4.png')

    def save_events(self):
        if self.hist:
            for ev in self.events:
                self.hist.Fill(ev.W, ev.Q2)
            self.hist.Draw()
            self.hist.c.Update()
        super().save_events()


if __name__=='__main__':
    EventGeneratorApp.launch()
