#!/usr/bin/python3
import numpy as np
import logging
logger = logging.getLogger(__name__)


class EventGeneratorBase:
    def __init__(self, conf):
        for f in 'wmin wmax q2min q2max ebeam channel events'.split():
            setattr(self, f, conf[f])
        # fixme
        self.dsigma_max = 1
        self.get_max_dsigma()

    def get_max_dsigma(self):
        if self.dsigma_max is None:
            self.dsigma_max = 5
        return self.dsigma_max

    def get_dsigma(self, W, Q2, cos_theta, phi):
        # TODO
        raise RuntimeEror("Not implemented")
        return 1

    def get_event(self, size=None):
        W  = np.random.uniform(self.wmin, self.wmax, size)
        Q2 = np.random.uniform(self.q2min, self.q2max, size)
        cos_theta = np.random.uniform(-1, 1, size)
        phi = np.random.uniform(0, 2*np.pi, size)
        return W, Q2, cos_theta, phi

    def generate_events(self):
        counter = self.events
        total = 0
        while counter > 0:
            total += 1
            #print('COUNTER: ', counter)
            W, Q2, cos_theta, phi = self.get_event()
            dsigma = self.get_dsigma(W, Q2, cos_theta, phi)
            if np.random.rand() < dsigma / self.dsigma_max:
                counter -= 1
                yield W, Q2, cos_theta, phi


class EventGeneratorApp:
    def __init__(self, EventGenerator, log_level=logging.INFO):
        logging.basicConfig(level=log_level)
        import yaml
        from pathlib import Path

        try:
            from dotenv import load_dotenv, find_dotenv
            load_dotenv(find_dotenv())
            import coloredlogs
            coloredlogs.install(
                level=log_level,
                fmt='%(asctime)s %(levelname)s %(message)s')
        except ModuleNotFoundError:
            pass

        import argparse

        parser = argparse.ArgumentParser(
            description='Event generator from helicity amplitudes.')
        parser.add_argument('--events', '-n', type=int,
            help='Number of events to generate')
        parser.add_argument('--ebeam', '-E', type=float,
            help='Beam energy, GeV')
        parser.add_argument('--q2min', type=float,
            help='Q^2 min, GeV^2')
        parser.add_argument('--q2max', type=float,
            help='Q^2 max, GeV^2')
        parser.add_argument('--wmin', type=float,
            help='W min, GeV')
        parser.add_argument('--wmax', type=float,
            help='W max, GeV')
        parser.add_argument('--channel', type=str,
            help='Channel')
        parser.add_argument('--output', '-o', type=str,
            default='wq2.dat',
            help='Output file name')
        self.args = parser.parse_args()

        with open('evgen.yaml') as f:
            self.conf = yaml.load(f)
        for attr in 'events ebeam channel wmin wmax q2min q2max'.split():
            if hasattr(self.args, attr) and getattr(self.args, attr, None) is not None:
                self.conf[attr] = getattr(self.args, attr)
        logger.info(self.conf)

        self.evgen = EventGenerator(self.conf)

    def run(self):
        #hist = Hists4()
        from estimate_time import EstimateTime
        timer = EstimateTime(self.evgen.events)
        timer.min_interval_to_output = 5  #  sec

        events = []
        for event in self.evgen.generate_events():
            events.append(event)

            timer.update()
            if timer.may_output():
                print("{:3.0f}%\tEvents: {}\tElapsed: {:8}\t Estimated: {:8}\tPer 1000 events: {:.3g}".format(
                    timer.percent, timer.counter,
                    timer.elapsed, timer.estimated,
                    (timer.elapsed_s / timer.counter)*1000
                ))

        print("Generated: {} events, time: {}".format(
            len(events), timer.elapsed))
        #hist.save()
        np.savetxt(self.args.output, events)
        logger.debug("Done")


if __name__=='__main__':

    #class EventGeneratorGauss:
        #def get_max_dsigma(self):

        #def get_dsigma(self, W, Q2, cos_theta, phi):

    EventGeneratorApp(
        EventGeneratorBase,
        log_level=logging.DEBUG
    ).run()
