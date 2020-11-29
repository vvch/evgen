#!/usr/bin/python3
import sys
import numpy as np
import scipy.interpolate
import logging
logger = logging.getLogger(__name__)

sys.path.append('/media/vitaly/work/work/jlab/clasfw')

import hep
import hep.amplitudes


class InterpSigma:
    def __init__(self, Amplitude, model, channel):
        data = Amplitude.query.filter_by(
            model=model,
            channel=channel,
        ).values(
            Amplitude.w,
            Amplitude.q2,
            Amplitude.cos_theta,
            # Amplitude.H1,
            Amplitude.H1r, Amplitude.H1j,
            Amplitude.H2r, Amplitude.H2j,
            Amplitude.H3r, Amplitude.H3j,
            Amplitude.H4r, Amplitude.H4j,
            Amplitude.H5r, Amplitude.H5j,
            Amplitude.H6r, Amplitude.H6j,
        )
        t = np.array(list(data))
        if not len(t):
            abort(404)  #  TODO: more friendly and informative error message

        t = t.T
        self.points = t[0:3].T

        # fixme: ugly temporary stub
        self.data = np.array([
            np.array([
                complex(*c)
                    for c in
                        zip(*[iter(p)] * 2)  #  pairwise
            ])
                for p in t[3:3+12].T ] )

        logger.info('Data loaded')
        logger.debug(f'SHAPE OF POINTS: {self.points.shape}')
        logger.debug(f'POINTS:\n{self.points}')
        logger.debug(f'SHAPE OF DATA: {self.data.shape}')
        logger.debug(f'DATA:\n{self.data}')


    def interp_H(self, w, q2, cos_theta):
        grid_w, grid_q2, grid_cθ = np.array(np.meshgrid(
            w, q2, cos_theta
        ))
        # )).transpose((0, 2, 1))

        grid_R = scipy.interpolate.griddata(
            self.points, self.data,
            (grid_w, grid_q2, grid_cθ),
            method='linear')
        return grid_R

    def interp_R(self, w, q2, cos_theta):
        grid_R = self.interp_H(w, q2, cos_theta)
        grid_R = np.apply_along_axis(
            hep.amplitudes.ampl_to_R, 3, grid_R,  #  3rd axis of grid_R with amplitudes
            # np.sum, 3, grid_R,
        )
        #grid_R = grid_R[:,:,:,self.qu_index]
        return grid_R

    def interp_dsigma(self, w, q2, cos_theta, phi, eps_T, h):
        def ampl_to_dsigma(H):
            return hep.amplitudes.H_to_dsigma(w, q2, eps_T, phi, h, H)
        grid_H = self.interp_H(w, q2, cos_theta)
        grid_dsig = np.apply_along_axis(
            ampl_to_dsigma, 3, grid_H,  #  3rd axis of grid_H with amplitudes
            # np.sum, 3, grid_R,
        )
        return grid_dsig


    def dsigma_minmax(self, Ebeam, h):
        raise RuntimeError("Not implemented yet")



class InterpSigmaLinearND(InterpSigma):
    def __init__(self, Amplitude, model, channel):
        super().__init__(Amplitude, model, channel)
        from scipy.interpolate.interpnd import _ndim_coords_from_arrays
        points = _ndim_coords_from_arrays(self.points)
        self.interpolator = scipy.interpolate.LinearNDInterpolator(
            points, self.data,
        )
        logger.info('Interpolator initialized')

    def interp_H(self, w, q2, cos_theta):
        grid_w, grid_q2, grid_cθ = np.array(np.meshgrid(
            w, q2, cos_theta
        ))
        # )).transpose((0, 2, 1))

        grid_H = self.interpolator(
            (grid_w, grid_q2, grid_cθ),
        )
        return grid_H


if __name__=="__main__":
    #logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv())
    except ModuleNotFoundError:
        pass
    try:
        import coloredlogs
        coloredlogs.install(
            #level=logging.DEBUG,
            level=logging.INFO,
            fmt='%(asctime)s %(levelname)s %(message)s',
                datefmt='%H:%M:%S')
    except ModuleNotFoundError:
        pass

    import argparse
    parser = argparse.ArgumentParser(
        description='Interpolated differential cross-section 3D plot from MAID helicity amplitudes data')
    parser.add_argument('-W', type=float,
        default=1.5,
        help='Final state invariant mass W, GeV')
    parser.add_argument('-Q2', type=float,
        default=1.0,
        help='Photon virtuality Q^2, GeV^2')
    parser.add_argument('--ebeam', '-E', type=float,
        default=10.6,
        help='Beam energy E, GeV')
    parser.add_argument('--channel', type=str,
        default='pi0 p',
        help='Channel')
    args = parser.parse_args()

    W  = args.W
    Q2 = args.Q2
    E_beam = args.ebeam
    channel_name = args.channel

    logger.info('Loading data')

    #from .models import Amplitude
    from clasfw.models import Amplitude, Model, Channel
    from clasfw.app import create_app
    app = create_app()
    ampl_pi0p = None
    channel = None
    with app.test_request_context():
        model = Model.by_name('maid')
        channel = Channel.by_name(channel_name)
        #ampl_pi0p = InterpSigma(Amplitude, model, channel)
        ampl_pi0p = InterpSigmaLinearND(Amplitude, model, channel)

        if 1:
            cos_theta=1
            print(f'W: {W},\tQ2: {Q2},\tcos_theta={cos_theta}')
            H = Amplitude.query.filter_by(
                model=model,
                channel=channel,
                w=W,
                q2=Q2,
                cos_theta=cos_theta,
            ).one().H
            print("H:\n", H)
            R = hep.amplitudes.ampl_to_R(H)
            print("R:\n", R)
            interp = ampl_pi0p.interp_R(W, Q2, cos_theta)
            print(interp)

    #cos_theta = np.linspace(-1, 1, 10)
    ε = 0.0000001
    cos_theta = np.arange(-1, 1 +ε, 0.01)
    #cos_theta = 0
    #cos_theta = 1
    grid_R = ampl_pi0p.interp_R(W, Q2, cos_theta )
    logger.debug(f'SHAPE OF R: {grid_R.shape}')
    logger.debug(f'R: {grid_R}')

    qu_index = 0
    grid_R = grid_R[:,:,:,qu_index]

    logger.debug(f'SHAPE OF R: {grid_R.shape}')
    logger.debug(f'R: {grid_R}')

    eps_T = hep.ε_T(W, Q2, E_beam)
    phi = np.deg2rad(np.linspace(0, 360, 120+1))

    grid_dsig = ampl_pi0p.interp_dsigma(W, Q2, cos_theta, phi, eps_T, h=1)
    logger.debug(grid_dsig.shape)

    import plotly.graph_objects as go
    fig = go.Figure(
        layout_scene_xaxis_title='φ, rad',
        layout_scene_yaxis_title='cos θ',
        layout_scene_zaxis_title='dσ/dΩ, µb',
        layout_title=f'{channel.name}: Q² = {Q2} GeV², W = {W} GeV',
    )
    fig.add_surface(
        x=phi,
        y=cos_theta,
        z=grid_dsig[0,0],
        hovertemplate=
            "φ: %{x} rad<br>"
            "cos<i>θ</i>: %{y}<br>"
            "dσ/dΩ: %{z} μb/sr"
            "<extra></extra>",
        colorscale='BlueRed',
    )
    #fig.add_trace(go.Scatter(
        #x=cos_theta,
        #y=grid_R.flatten(),
    #))
    fig.show()
