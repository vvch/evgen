#!/usr/bin/python3

import sys
import numpy as np


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
            print('COUNTER: ', counter)
            W, Q2, cos_theta, phi = self.get_event()
            dsigma = self.get_dsigma(W, Q2, cos_theta, phi)
            if np.random.rand() < dsigma / self.dsigma_max:
                counter -= 1
                yield W, Q2, cos_theta, phi


if __name__=='__main__':
    from pathlib import Path
    import yaml

    with open(Path(__file__).with_suffix('.yaml')) as f:
        conf = yaml.load(f)
    print(conf)

    evg = EventGeneratorBase(conf)
    ev = list(evg.generate_events())
    print("\n".join(str(e)for e in ev))
    print('LEN: ', len(ev))
