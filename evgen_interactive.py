#!/usr/bin/python3
import numpy as np
import logging
logger = logging.getLogger(__name__)

from evgen import EventGeneratorBase, EventGeneratorApp, EventGeneratorFW
from hist import Hists4


class EventGeneratorApp(EventGeneratorApp):
    def run(self):
        from estimate_time import EstimateTime
        timer = EstimateTime(self.evgen.events)
        timer.min_interval_to_output = self.args.interval
        hist = Hists4()
        hist.CreateCanvas()
        hist.Draw()

        events = []
        for event in self.evgen.generate_events():
            events.append(event)
            W, Q2, cos_theta, phi = event
            hist.Fill(W, Q2)
            timer.update()
            if timer.may_output():
                events = []
                hist.Draw()
                hist.c.Update()
                logger.info(
                    "{:3.0f}%\tEvents: {}\tElapsed: {:8}\t Estimated: {:8}\tSpeed: {:3g}/min"
                    .format(
                        timer.percent, timer.counter,
                        timer.elapsed, timer.estimated,
                        timer.speed*60
                ))

        #logger.info("Generated: {} events, time: {}".format(
            #len(events), timer.elapsed))
        hist.save('events_hist4.png')
        #np.savetxt(self.args.output, events)
        logger.debug("Done")


if __name__=='__main__':
    import sys
    try:
        EventGeneratorApp(
            EventGeneratorFW,
            log_level=logging.INFO
        ).run()
    except (NotImplementedError, ModuleNotFoundError) as e:
        logger.fatal(e)
        sys.exit(1)
