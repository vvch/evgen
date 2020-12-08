#!/usr/bin/python3
from evgen_interactive import EventGeneratorBase, EventGeneratorApp
from numpy import sin, cos


class EventGeneratorTest(EventGeneratorBase):
    def get_dsigma(self, event):
        return sin(event.Q2*3)*cos(event.W*3)

    def get_dsigma_upper(self):
        return 1


EventGeneratorApp.launch(EventGeneratorTest)
