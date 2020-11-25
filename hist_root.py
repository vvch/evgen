#!/usr/bin/python3

import sys
import logging as logger
import numpy as np
from ROOT import TCanvas, TH1F, TH2F

#np.random.seed(1)


W_min = 1.08
W_max = 2
Q2_min = 0
Q2_max = 5

W_margin = 0.005
Q2_margin = 0.025
W_bins = 186
Q2_bins = 202


class Hists4:
    def __init__(self):
        self.c1 = TCanvas("c1", "Histogram",
            1280, 1080)
        self.c1.Divide(2,2)

        self.h_WQ2 = TH2F("h1", "2D Histogram (W,Q^{2})",
            W_bins, W_min-W_margin,   W_max+W_margin,
            Q2_bins, Q2_min-Q2_margin, Q2_max+Q2_margin)
        self.h_W = TH1F("h3", "Histogram W",
            W_bins, W_min, W_max)
        self.h_Q2 = TH1F("h4", "Histogram Q^{2}",
            Q2_bins, Q2_min, Q2_max)

    def Fill(self, W, Q2):
        self.h_WQ2.Fill(W, Q2)
        self.h_W.Fill(W)
        self.h_Q2.Fill(Q2)

    def Draw(self):
        self.c1.cd(1)
        self.h_WQ2.Draw("COL")
        self.c1.cd(2)
        self.h_WQ2.Draw("SURF2")
        self.c1.cd(3)
        self.h_W.Draw()
        self.c1.cd(4)
        self.h_Q2.Draw()

    def save(self, fname="histograms4.png"):
        self.Draw()
        self.c1.Print(fname)


def norm(c1, c2):
    mean   = (c1 + c2) / 2
    stddev = (c2 - c1) / 4
    return np.random.normal(mean, stddev)


if __name__ =='__main__':

    hists = Hists4()
    data = np.loadtxt('wq2.dat')
    #data = np.loadtxt(sys.stdin)
    #W, Q2 = data[0:2]
    #hists.Fill(W, Q2)
    for r in data:
        W, Q2 = r[0:2]
        hists.Fill(W, Q2)

    hists.save("hist4.png")
"""
    for ev in range(1000000):
        W  = norm(W_min, W_max)
        Q2 = norm(Q2_min, Q2_max)
        hists.Fill(W, Q2)
    hists.save()
"""
