#!/usr/bin/python3
import sys
import numpy as np
from ROOT import TCanvas, TH1D, TH2D, gROOT


W_min  = 1.08
W_max  = 2
Q2_min = 0
Q2_max = 5

W_margin  = 0.005
Q2_margin = 0.025
W_bins  = 186
Q2_bins = 202


class Hists4:
    def __init__(self):
        self.h_WQ2 = TH2D("wq2", ";W, GeV;Q^{2}, GeV^{2}",
            W_bins,  W_min-W_margin,   W_max+W_margin,
            Q2_bins, Q2_min-Q2_margin, Q2_max+Q2_margin)
        self.h_W = TH1D("w", ";W, GeV",
            W_bins, W_min, W_max)
        self.h_Q2 = TH1D("q2", ";Q^{2}, GeV^{2}",
            Q2_bins, Q2_min, Q2_max)
        self.Fill = np.vectorize(self.FillScalar)

    def FillScalar(self, W, Q2):
        self.h_WQ2.Fill(W, Q2)
        self.h_W.Fill(W)
        self.h_Q2.Fill(Q2)

    def CreateCanvas(self):
        try:
            self.c
        except AttributeError:
            self.c = TCanvas("c", "Events Histograms",
                1280, 1080)
            self.c.Divide(2, 2)

    def Draw(self):
        self.c.cd(1)
        self.h_WQ2.Draw("COL")
        self.c.cd(2)
        self.h_WQ2.Draw("SURF2")
        self.c.cd(3)
        self.h_W.Draw()
        self.c.cd(4)
        self.h_Q2.Draw()

    def save(self, fname):
        gROOT.SetBatch(True)
        self.CreateCanvas()
        self.Draw()
        self.c.Print(fname)

    @classmethod
    def produce(cls, W, Q2, fname):
        hists = cls()
        hists.Fill(W, Q2)
        hists.save(args.output)


def norm(x1, x2, size=None):
    mean   = (x1 + x2) / 2
    stddev = (x2 - x1) / 4
    return np.random.normal(mean, stddev, size=size)


if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Plotting events histograms by CERN ROOT')
    parser.add_argument('file', type=str, nargs='?', default='wq2.dat',
        help='Events table file name')
    parser.add_argument('--output', '-o', type=str, default='events_hist4.png',
        help='Output file name')
    parser.add_argument('--test', type=int, nargs='?', metavar='N', const=100000,
        help='Use Gauss distribution of N events instead of events file (for test only)')
    args = parser.parse_args()

    if args.test:
        W  = norm(W_min,  W_max,  args.test)
        Q2 = norm(Q2_min, Q2_max, args.test)
    else:
        W, Q2 = np.loadtxt(args.file, unpack=True, usecols=(0,1))
    Hists4.produce(W, Q2, args.output)
