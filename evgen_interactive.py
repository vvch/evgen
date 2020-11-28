#!/usr/bin/python3
import numpy as np
import logging
logger = logging.getLogger(__name__)

from evgen import EventGeneratorApp, EventGeneratorMAID
from hist import Hists4


class EventGeneratorApp(EventGeneratorApp):
    def run(self):
        from estimate_time import EstimateTime
        timer = EstimateTime(self.evgen.events)
        timer.min_interval_to_output = 1  #  sec
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
                print("{:3.0f}%\tEvents: {}\tElapsed: {:8}\t Estimated: {:8}\tPer 1000 events: {:.3g}".format(
                    timer.percent, timer.counter,
                    timer.elapsed, timer.estimated,
                    (timer.elapsed_s / timer.counter)*1000
                ))

        #print("Generated: {} events, time: {}".format(
            #len(events), timer.elapsed))
        hist.save('events_hist4.png')
        #np.savetxt(self.args.output, events)
        logger.debug("Done")


if __name__=='__main__':

    try:
        EventGeneratorApp(
            EventGeneratorMAID,
            log_level=logging.INFO
        ).run()
    except (NotImplementedError, ModuleNotFoundError) as e:
        logger.fatal(e)
        sys.exit(1)
