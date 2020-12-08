#!/usr/bin/python3
import sys
import numpy as np
from collections import namedtuple
import logging
logger = logging.getLogger(__name__)

__author__ = "Vitaly Chesnokov"


Event = namedtuple('Event',
    ('W', 'Q2', 'cos_theta', 'phi'))


class EventGeneratorBase:
    """Abstract base event generator"""
    def __init__(self, conf):
        for f in 'wmin wmax q2min q2max ebeam channel events'.split():
            setattr(self, f, conf[f])
        self.min_dsigma = None
        self.max_dsigma = None
        self.max_dsigma_event = None
        self.dsigma_exceed_counter = 0
        self.raw_events_counter = 0
        try:
            self.dsigma_upper = conf['dsigmaupper']
        except KeyError:
            self.dsigma_upper = self.get_dsigma_upper()

    def get_dsigma_upper(self):
        raise NotImplementedError(
            "Currently automatic calculation of maximum differential cross-section"
            " value in the specified range is not implemented,"
            " it should be specified as 'dsigmaupper' parameter")

    def get_dsigma(self, event):
        raise NotImplementedError(
            "No differential cross-section calculation method provided")

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
            dsigma = self.get_dsigma(ev)
            if self.max_dsigma is None or self.max_dsigma < dsigma:
                self.max_dsigma = dsigma
                self.max_dsigma_event = ev
            if self.min_dsigma is None or self.min_dsigma > dsigma:
                self.min_dsigma = dsigma
            if dsigma > self.dsigma_upper:
                if not self.dsigma_exceed_counter:
                    logger.warning(
                        "Cross-section %g exceeded upper limit %g mcb for %s."
                        " Upper limit may be specified incorrectly.",
                        dsigma, self.dsigma_upper, str(ev))
                self.dsigma_exceed_counter +=1
            if np.random.rand() < dsigma / self.dsigma_upper:
                counter -= 1
                yield ev


class EventGeneratorApp:
    """Event Generator"""
    def __init__(self, EventGenerator, log_level=logging.INFO, log_fmt='%(asctime)s %(levelname)s %(message)s'):
        logging.basicConfig(level=log_level, format=log_fmt, datefmt='%H:%M:%S')
        import yaml
        from pathlib import Path
        try:
            from dotenv import load_dotenv, find_dotenv
            load_dotenv(find_dotenv())
        except ModuleNotFoundError:
            pass
        try:
            import coloredlogs
            coloredlogs.install(level=log_level, fmt=log_fmt, datefmt='%H:%M:%S')
        except ModuleNotFoundError:
            pass

        import argparse
        self.parser = argparse.ArgumentParser(
            fromfile_prefix_chars='@',
            description=EventGenerator.__doc__ or self.__doc__ or EventGeneratorApp.__doc__)
        self.parser.add_argument('--events', '-n', '-N', type=int,
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
        self.parser.add_argument('--dsigmaupper', '-U', type=float,
            help='Upper limit for differential cross-section value, mcb')
        self.parser.add_argument('--channel', '-C', type=str,
            choices=['pi+ n', 'pi0 p', 'pi- p', 'pi0 n'],
            help='Channel')
        self.parser.add_argument('--interval', '-T', type=float,
            default=1,
            help='Output time interval, seconds')
        self.parser.add_argument('--output', '-o', type=str,
            default='wq2.dat',
            help='Output file name')
        self.args = self.parser.parse_args()

        with open('evgen.yaml') as f:
            self.conf = yaml.load(f)
        for attr in 'events ebeam channel wmin wmax q2min q2max dsigmaupper'.split():
            if hasattr(self.args, attr) and getattr(self.args, attr, None) is not None:
                self.conf[attr] = getattr(self.args, attr)
        logger.info(self.conf)

        self.evgen = EventGenerator(self.conf)

    def run(self):
        #hist = Hists4()
        from estimate_time import EstimateTime
        timer = EstimateTime(self.evgen.events)
        timer.min_interval_to_output = self.args.interval

        events = []
        for event in self.evgen.generate_events():
            events.append(event)

            timer.update()
            if timer.may_output():
                logger.info(
                    "%3.0f %%\tEvents: %d\tElapsed: %8s\t Estimated: %8s\tSpeed: %3g/min",
                        timer.percent, timer.counter,
                        timer.elapsed, timer.estimated,
                        timer.speed * 60)

        logger.info("Generated: %d events, time: %s", len(events), timer.elapsed)
        logger.info(
            "Filtered %d differential cross-section values at all: min=%g, max=%g [mcb]",
            self.evgen.raw_events_counter,
            self.evgen.min_dsigma, self.evgen.max_dsigma)
        if self.evgen.dsigma_exceed_counter:
            logger.warning(
                "Cross-section %d times (of %d, %.3g%%) exceeded upper limit %g, max=%g [mcb] on %s",
                self.evgen.dsigma_exceed_counter, self.evgen.raw_events_counter,
                self.evgen.dsigma_exceed_counter / self.evgen.raw_events_counter,
                self.evgen.dsigma_upper, self.evgen.max_dsigma,
                str(self.evgen.max_dsigma_event))
        #hist.save()
        np.savetxt(self.args.output, events)
        logger.debug("Done")

    @classmethod
    def launch(cls, evgenclass, log_level=logging.INFO):
        try:
            cls(evgenclass, log_level).run()
        except (NotImplementedError, ModuleNotFoundError) as e:
            logger.fatal(e)
            sys.exit(1)


class EventGeneratorTest(EventGeneratorBase):
    """Test event generator with simple distribution function instead of real cross-section"""
    def get_dsigma(self, event):
        return np.sin(event.Q2*3)*np.cos(event.W*3)
    def get_dsigma_upper(self):
        return 1


if __name__=='__main__':
    EventGeneratorApp.launch(EventGeneratorTest, logging.DEBUG)
