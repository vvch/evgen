#!/usr/bin/python3
import numpy as np
import logging
logger = logging.getLogger(__name__)
from collections import namedtuple


Event = namedtuple('Event',
    ('W', 'Q2', 'cos_theta', 'phi'))


class EventGeneratorBase:
    """Event generator"""
    def __init__(self, conf):
        for f in 'wmin wmax q2min q2max ebeam channel events'.split():
            setattr(self, f, conf[f])
        self.min_dsigma = None
        self.max_dsigma = None
        self.max_dsigma_event = None
        self.exceed_dsigma_counter = 0
        self.raw_events_counter = 0
        try:
            self.dsigma_max = conf['dsigmamax']
        except KeyError:
            self.dsigma_max = self.get_max_dsigma()

    def get_max_dsigma(self):
        raise NotImplementedError(
            "Currently automatic calculation of maximum differential cross-section value in the specified range is not implemented, it should be specified as 'dsigmamax' parameter")

    def get_dsigma(self, W, Q2, cos_theta, phi):
        raise NotImplementedError(
            "No cross-section calculation method provided")

    def raw_event(self, size=None):
        return Event(
            W         = np.random.uniform(self.wmin,  self.wmax,  size),
            Q2        = np.random.uniform(self.q2min, self.q2max, size),
            cos_theta = np.random.uniform(-1, 1, size),
            phi       = np.random.uniform(0, 2*np.pi, size),
        )

    def generate_events(self):
        counter = self.events
        while counter > 0:
            self.raw_events_counter += 1
            #log.debug('COUNTER: ', counter)
            ev = self.raw_event()
            dsigma = self.get_dsigma(*ev)
            if self.max_dsigma is None or self.max_dsigma < dsigma:
                self.max_dsigma = dsigma
                self.max_dsigma_event = ev
            if self.min_dsigma is None or self.min_dsigma > dsigma:
                self.min_dsigma = dsigma
            if dsigma > self.dsigma_max:
                if not self.exceed_dsigma_counter:
                    logger.warning(
                        f"Cross-section {dsigma} exceeded upper limit {self.dsigma_max} mcb for {ev}. Upper limit may be specified incorrectly.")
                self.exceed_dsigma_counter +=1
            if np.random.rand() < dsigma / self.dsigma_max:
                counter -= 1
                yield ev


class EventGeneratorApp:
    description = "Event Generator"
    def __init__(self, EventGenerator, log_level=logging.INFO):
        logging.basicConfig(level=log_level)
        import yaml
        from pathlib import Path
        try:
            from dotenv import load_dotenv, find_dotenv
            load_dotenv(find_dotenv())
        except ModuleNotFoundError:
            pass
        try:
            import coloredlogs
            coloredlogs.install(
                level=log_level,
                fmt='%(asctime)s %(levelname)s %(message)s',
                datefmt='%H:%M:%S')
        except ModuleNotFoundError:
            pass

        import argparse
        self.parser = argparse.ArgumentParser(
            description=self.description)
        self.parser.add_argument('--events', '-n', type=int,
            help='Number of events to generate')
        self.parser.add_argument('--ebeam', '-E', type=float,
            help='Beam energy, GeV')
        self.parser.add_argument('--wmin', type=float,
            help='W min, GeV')
        self.parser.add_argument('--wmax', type=float,
            help='W max, GeV')
        self.parser.add_argument('--q2min', type=float,
            help='Q^2 min, GeV^2')
        self.parser.add_argument('--q2max', type=float,
            help='Q^2 max, GeV^2')
        self.parser.add_argument('--dsigmamax', type=float,
            help='Maximum differential cross-section value, mcb')
        self.parser.add_argument('--channel', '-C', type=str,
            choices=['pi+ n', 'pi0 p', 'pi- p', 'pi0 n'],
            help='Channel')
        self.parser.add_argument('--output', '-o', type=str,
            default='wq2.dat',
            help='Output file name')
        self.args = self.parser.parse_args()

        with open('evgen.yaml') as f:
            self.conf = yaml.load(f)
        for attr in 'events ebeam channel wmin wmax q2min q2max dsigmamax'.split():
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
                print("{:3.0f}%\tEvents: {}\tElapsed: {:8}\t Estimated: {:8}\tSpeed: {:3g}/min".format(
                    timer.percent, timer.counter,
                    timer.elapsed, timer.estimated,
                    timer.speed * 60,
                ))

        print("Generated: {} events, time: {}".format(
            len(events), timer.elapsed))
        logger.info(
            f"Filtered {self.evgen.raw_events_counter} differential cross-section values at all: min={self.evgen.min_dsigma}, max={self.evgen.max_dsigma}, [mcb]")
        if self.evgen.exceed_dsigma_counter:
            logger.warning(
                f"Cross-section {self.evgen.exceed_dsigma_counter} times (of {self.evgen.raw_events_counter}) exceeded upper limit {self.evgen.dsigma_max}, max={self.evgen.max_dsigma} mcb on {self.evgen.max_dsigma_event}")
        #hist.save()
        np.savetxt(self.args.output, events)
        logger.debug("Done")


if __name__=='__main__':
    import sys

    class EventGeneratorTest(EventGeneratorBase):
        """Test event generator with simple distribution function instead of real cross-section"""
        def get_dsigma(self, W, Q2, cos_theta, phi):
            return np.sin(Q2*3)*np.cos(W*3)
        def get_max_dsigma(self):
            return 1

    try:
        EventGeneratorApp(
            EventGeneratorTest,
            log_level=logging.DEBUG
        ).run()
    except (NotImplementedError, ModuleNotFoundError) as e:
        logger.fatal(e)
        sys.exit(1)
