#!/usr/bin/python3
import numpy as np
from ROOT import TLorentzVector, TVector3


class Lund:
    e  = 11
    γ  = 22
    π0 = 111
    πp = 211
    πm = -211
    Kp = 321
    n  = 2112
    p  = 2212
    Λ  = 3122
    Σ0 = 3212

    available_channels = ('pi+ n', 'pi0 p', 'pi- p', 'pi0 n')

    def __init__(self, channel_name, ebeam):
        if channel_name not in self.available_channels:
            NotImplementedError(
                f'Sorry, currently LUND file saving implemented for'
                f' {self.available_channels} channels only')
        self.channel_name = channel_name
        self.ebeam = ebeam

    def event(self, event):
        W, Q2, cos_θ, φ = event
        E0 = self.ebeam

        M_p  = 0.9382720813  #  [GeV], Proton mass
        M_π  = 0.13957018    #  [GeV], Charged pion mass
        M_π0 = 0.1349766     #  [GeV], Neutral pion mass
        M_e  = 0.0005        #  [GeV], Electron mass

        if self.channel_name == 'pi+ n':
            mpi = M_π
            mp  = M_p
            lund_target = Lund.p
            lund_meson  = Lund.πp
            lund_baryon = Lund.n
        elif self.channel_name == 'pi0 p':
            mpi = M_π0
            mp  = M_p
            lund_target = Lund.p
            lund_meson  = Lund.π0
            lund_baryon = Lund.p
        elif self.channel_name == 'pi- p':
            mpi = M_π
            mp  = M_p  #  fixme: should be M_n
            lund_target = Lund.n
            lund_meson  = Lund.πm
            lund_baryon = Lund.p
        elif self.channel_name == 'pi0 n':
            mpi = M_π0
            mp  = M_p  #  fixme: should be M_n
            lund_target = Lund.n
            lund_meson  = Lund.π0
            lund_baryon = Lund.n

        θ     = np.arccos(cos_θ)
        sin_θ = np.sin(θ)
        cos_φ = np.cos(φ)
        sin_φ = np.sin(φ)

        Epi = (W*W + mpi*mpi - mp*mp)/(2*W)
        Ep  = (W*W + mp*mp - mpi*mpi)/(2*W)
        p   = np.sqrt(Epi*Epi - mpi*mpi)
        nu  = (W*W + Q2 - mp*mp)/(2*mp)

        ang1 = np.deg2rad(np.random.uniform(0, 180))
        ang2 = np.deg2rad(np.random.uniform(0, 360))
        sin_ang1 = np.sin(ang1)
        cos_ang1 = np.cos(ang1)
        sin_ang2 = np.sin(ang2)
        cos_ang2 = np.cos(ang2)

        nucleon = TLorentzVector(
            -p*cos_φ*sin_θ,
            -p*sin_φ*sin_θ,
            -p*cos_θ,
            Ep)
        meson = TLorentzVector(
            p*cos_φ*sin_θ,
            p*sin_φ*sin_θ,
            p*cos_θ,
            Epi)
        γ1 = TLorentzVector(
            mpi*cos_ang2*sin_ang1/2,
            mpi*sin_ang2*sin_ang1/2,
            mpi*cos_ang1/2,
            mpi/2)
        γ2 = TLorentzVector(
            -mpi*cos_ang2*sin_ang1/2,
            -mpi*sin_ang2*sin_ang1/2,
            -mpi*cos_ang1/2,
            mpi/2)

        β = TVector3(
            0,
            0,
            p/Epi)

        γ1.Boost(β)
        γ2.Boost(β)

        γ1.RotateY(θ)
        γ2.RotateY(θ)
        γ1.RotateZ(φ)
        γ2.RotateZ(φ)

        β.SetXYZ(
            0,
            0,
            np.sqrt(nu*nu + Q2)/(nu + mp))

        for p in (nucleon, meson, γ1, γ2):
            p.Boost(β)

        tmp_ang = np.arccos(
            (Q2 + 2*E0*nu) /
            (2*E0*np.sqrt(nu*nu + Q2))
        )
        for p in (nucleon, meson, γ1, γ2):
            p.RotateY(tmp_ang)

        e = TLorentzVector(
            (E0 - nu)*np.sqrt(1 - (1 - Q2/(2*E0*(E0 - nu)))**2),
            0,
            (E0 - nu)*(1 - Q2/(2*E0*(E0 - nu))),
            E0 - nu)

        ang2 = np.deg2rad(np.random.uniform(0, 360))

        for p in (nucleon, meson, γ1, γ2, e):
            p.RotateZ(ang2)

        def str_p4(p):
            return ' '.join([str(p[i]) for i in range(4)])

        e, γ1, γ2, nucleon, meson = map(str_p4, (
            e, γ1, γ2, nucleon, meson))

        if 0:  ##  pi0p + pi0 decay
            lund = f"4 0 0 0 0 {Lund.e} {E0} {lund_target} 0 0"    "\n"\
                 + f"0 0 0 {Lund.e} 0 0 {e} {M_e} 0 0 0"           "\n"\
                 + f"0 0 0 {Lund.p} 0 0 {nucleon} {mp} 0 0 0"      "\n"\
                 + f"0 0 0 {Lund.γ} 0 0 {γ1} 0 0 0 0"              "\n"\
                 + f"0 0 0 {Lund.γ} 0 0 {γ2} 0 0 0 0"              "\n"
        else:  ##  pi0p, pi+n, pi-p, pi0n
            lund = f"3 0 0 0 0 {Lund.e} {E0} {lund_target} 0 0"    "\n"\
                 + f"0 0 0 {Lund.e} 0 0 {e} {M_e} 0 0 0"           "\n"\
                 + f"0 0 0 {lund_baryon} 0 0 {nucleon} {mp} 0 0 0" "\n"\
                 + f"0 0 0 {lund_meson} 0 0 {meson} {mpi} 0 0 0"   "\n"
        #lund = lund.format(**vars())
        return lund


if __name__ =='__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(
        description='LUND format conversion')
    parser.add_argument('file', type=str, nargs='?', default='wq2.dat',
        help='Events table file name')
    parser.add_argument('--channel', '-C', type=str, required=True,
        choices=Lund.available_channels,
        help='Channel')
    parser.add_argument('--ebeam', '-E', type=float, required=True,
        help='Beam energy, GeV')
    parser.add_argument('--output', '-o', type=argparse.FileType('w'),
        default=sys.stdout,
        help='Output file name')
    args = parser.parse_args()

    lund = Lund(args.channel, args.ebeam)
    events = np.loadtxt(args.file)
    for ev in (events):
        #print(ev)
        print(lund.event(ev), end='', file=args.output)
