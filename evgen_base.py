#!/usr/bin/python3
"""
Base event generator framework, independent of cross-section calculation method
"""
import sys, time
import re
import numpy as np
from collections import namedtuple
import logging
logger = logging.getLogger(__name__)

__author__ = "Vitaly Chesnokov"


Event = namedtuple('Event',
    ('W', 'Q2', 'cos_theta', 'phi'))


class EventGeneratorBase:
    """Abstract base event generator"""
    __slots__ = """
        wmin   wmax
        q2min  q2max
        ctmin  ctmax
        phimin phimax
        min_dsigma
        max_dsigma
        max_dsigma_event
        dsigma_upper
        dsigma_exceed_counter
        raw_events_counter
        events
    """.split()

    def __init__(self, conf):
        self.min_dsigma = None
        self.max_dsigma = None
        self.max_dsigma_event = None
        self.dsigma_exceed_counter = 0
        self.raw_events_counter = 0
        for k, v in conf.__dict__.items():
            if k in ('phimin', 'phimax'):
                v = np.deg2rad(v)
            setattr(self, k, v)

        if getattr(self, 'dsigma_upper', None) is None:
            self.dsigma_upper = self.get_dsigma_upper()

    def get_dsigma_upper(self):
        raise NotImplementedError(
            "Currently automatic calculation of maximum differential cross-section"
            " value in the specified range is not implemented,"
            " it should be specified as 'dsigma-upper' parameter")

    def get_dsigma(self, event):
        raise NotImplementedError(
            "No differential cross-section calculation method provided")

    def raw_event(self, size=None):
        return Event(
            W         = np.random.uniform(self.wmin,   self.wmax,   size),
            Q2        = np.random.uniform(self.q2min,  self.q2max,  size),
            cos_theta = np.random.uniform(self.ctmin,  self.ctmax,  size),
            phi       = np.random.uniform(self.phimin, self.phimax, size),
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
                        "Cross-section %g exceeded upper limit %g [mcb*GeV^-3] for %s."
                        " Upper limit may be specified incorrectly.",
                        dsigma, self.dsigma_upper, str(ev))
                self.dsigma_exceed_counter +=1
            if np.random.rand() < dsigma / self.dsigma_upper:
                counter -= 1
                yield ev


class EventGeneratorApp:
    """Event Generator"""
    def __init__(self, EventGenerator, log_level=logging.INFO,
                 log_fmt='%(asctime)s %(levelname)s %(message)s'):
        logging.basicConfig(level=log_level, format=log_fmt, datefmt='%H:%M:%S')
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

        from configargparse import ArgumentParser, DefaultsFormatter
        parser = ArgumentParser(
            fromfile_prefix_chars='@',
            default_config_files=['evgen.conf'],
            formatter_class=DefaultsFormatter,
            add_config_file_help=False,
            description=EventGenerator.__doc__
                or self.__doc__ or EventGeneratorApp.__doc__)
        parser.add('-c', '--config', is_config_file=True,
            help='Config file path')
        parser.add('--events', '-n', '-N', type=int,
            help='Number of events to generate')
        parser.add('--ebeam', '-E', type=float,
            help='Beam energy, GeV')
        parser.add('--wmin', type=float, default=1.1,
            help='W min, GeV')
        parser.add('--wmax', type=float, default=2,
            help='W max, GeV')
        parser.add('--q2min', type=float, default=0,
            help='Q^2 min, GeV^2')
        parser.add('--q2max', type=float, default=5,
            help='Q^2 max, GeV^2')
        parser.add('--ctmin', '--cos-theta-min', type=float, default=-1,
            help='cos theta min')
        parser.add('--ctmax', '--cos-theta-max', type=float, default=+1,
            help='cos theta max')
        parser.add('--phimin', type=float, default=0,
            help='phi min, deg')
        parser.add('--phimax', type=float, default=360,
            help='phi max, deg')
        parser.add('--dsigma-upper', '-U', type=float,
            help='Upper limit for differential cross-section value, mcb')
        parser.add('--helicity', '-H', type=int, default=0,
            choices=[-1, 0, 1],
            help='Electron helicity (use 0 for random choice)')
        parser.add('--channel', '-C', type=str, required=True,
            choices=['pi+ n', 'pi0 p', 'pi- p', 'pi0 n'],
            help='Channel')
        parser.add('--interval', '-T', type=float, default=1,
            help='Output time interval, seconds')
        parser.add('--output', '-o', type=str, default='wq2.dat',
            help='Output file name')
        self.parser = parser
        self.args = parser.parse()
        self.evgen = EventGenerator(self.args)
        print(self.get_header(), end=None)
        logger.info(self.args)

    def get_header(self):
        def format_range(min, max, sep=' - '):
            return "{:<4}".format(min) if min == max  \
              else "{:<4}{}{:<4}".format(min, sep, max)
        a = self.args
        h = a.helicity
        h = f"{h:+}" if h        \
            else 'random +1 or -1'
        dsigma_upper_type = "manually specified"  \
            if a.dsigma_upper is not None         \
            else "calculated"
        return (
            self.parser.description + "\n"
            f"Author: {__author__}\n"
            f"\n"
            f"Started:         {time.asctime()}\n"
            f"Channel:         {a.channel}\n"
            f"W  range:        {format_range(a.wmin,   a.wmax)} GeV\n"
            f"Q² range:        {format_range(a.q2min,  a.q2max)} GeV²\n"
            f"cos θ range:     {format_range(a.ctmin,  a.ctmax)}\n"
            f"φ range:         {format_range(a.phimin, a.phimax)} degrees\n"
            f"E beam:          {a.ebeam} GeV\n"
            f"Helicity:        {h}\n"
            f"DCS upper limit: {self.evgen.dsigma_upper} µb·GeV⁻³ ({dsigma_upper_type})\n"
            f"Events number:   {a.events}\n"
        )

    def get_header_commented(self):
        return re.sub(r'^', '#  ', '\n'+self.get_header()+'\n', 0, re.M) + '\n'

    def get_footer(self, timer):
        return (
            f"Generated:       {timer.counter} events\n"
            f"Elapsed time:    {timer.elapsed}\n"
            f"Filtered events: {self.evgen.raw_events_counter}\n"
            f"Filtered rate:   {timer.counter / self.evgen.raw_events_counter:%}\n"
            f"DCS min:         {self.evgen.min_dsigma} µb·GeV⁻³\n"
            f"DCS max:         {self.evgen.max_dsigma} µb·GeV⁻³\n"
            f"DCS upper limit: {self.evgen.dsigma_upper} µb·GeV⁻³\n"
            f"DCS max/limit:   {self.evgen.max_dsigma / self.evgen.dsigma_upper:%}\n"
            f"DCS lim exceed:  {self.evgen.dsigma_exceed_counter}\n"
            f"Finished:        {time.asctime()}\n"
        )

    def get_footer_commented(self, timer):
        return '#\n' + re.sub(r'^', '#  ', self.get_footer(timer), 0, re.M)

    def run(self):
        #hist = Hists4()
        from estimate_time import EstimateTime
        timer = EstimateTime(self.evgen.events)
        timer.min_interval_to_output = self.args.interval

        events = []
        with open(self.args.output, 'w') as output:
            output.write(self.get_header_commented())
            for event in self.evgen.generate_events():
                events.append(event)

                timer.update()
                if timer.may_output():
                    logger.info(
                        "%3.0f %%\tEvents: %d\tElapsed: %8s\t Estimated: %8s\tSpeed: %3g/min",
                            timer.percent, timer.counter,
                            timer.elapsed, timer.estimated,
                            timer.speed * 60)
                    if 1:
                        np.savetxt(output, events)
                        events.clear()

            np.savetxt(output, events)
            #hist.save()
            output.write(self.get_footer_commented(timer))
            logger.info(
                "Generated: %d events, elapsed time: %s",
                timer.counter, timer.elapsed)
            logger.info(
                "Filtered %d differential cross-section values"
                " at all: min=%g, max=%g [mcb*GeV^-3]",
                self.evgen.raw_events_counter,
                self.evgen.min_dsigma, self.evgen.max_dsigma)
            if self.evgen.dsigma_exceed_counter:
                logger.warning(
                    "Cross-section %d times (of %d, %.3g%%) exceeded upper limit %g,"
                    " max=%g [mcb*GeV^-3] on %s",
                    self.evgen.dsigma_exceed_counter, self.evgen.raw_events_counter,
                    self.evgen.dsigma_exceed_counter / self.evgen.raw_events_counter,
                    self.evgen.dsigma_upper, self.evgen.max_dsigma,
                    str(self.evgen.max_dsigma_event))
        logger.debug("Done")

    @classmethod
    def launch(cls, evgenclass, log_level=logging.INFO):
        try:
            cls(evgenclass, log_level).run()
        except (NotImplementedError, ModuleNotFoundError) as e:
            logger.fatal(e)
            sys.exit(1)
        except KeyboardInterrupt as e:
            logger.fatal(e.__class__.__name__)
            sys.exit(2)


class EventGeneratorTest(EventGeneratorBase):
    """Test event generator with simple distribution function instead of real cross-section"""
    def get_dsigma(self, event):
        return np.sin(event.Q2*3)*np.cos(event.W*3)
    def get_dsigma_upper(self):
        return 1


if __name__=='__main__':
    EventGeneratorApp.launch(EventGeneratorTest, logging.DEBUG)
