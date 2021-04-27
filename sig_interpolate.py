#!/usr/bin/python3
import sys
import numpy as np
import scipy.interpolate
import logging
logger = logging.getLogger(__name__)

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
        ).order_by(
            Amplitude.q2,
            Amplitude.w,
            Amplitude.cos_theta,
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


    def correct_maid_H56(self):
        logger.info('Correct MAID data')
        for i in range(len(self.points)):
            W, Q2 = self.points[i, 0:2]
            self.data[i, 4:6] *= hep.amplitudes.H56_maid_correction_factor(W, Q2)  ##  H5, H6


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

    def interp_dsigma_eps(self, w, q2, cos_theta, phi, eps_T, h=1):
        def ampl_to_dsigma(H):
            return hep.amplitudes.H_to_dsigma(w, q2, eps_T, phi, h, H)
        grid_H = self.interp_H(w, q2, cos_theta)
        grid_dsig = np.apply_along_axis(
            ampl_to_dsigma, 3, grid_H,  #  3rd axis of grid_H with amplitudes
            # np.sum, 3, grid_R,
        )
        return grid_dsig

    def interp_dsigma_v(self, W, Q2, cos_theta, phi, Eb, h=1):
        return self.interp_dsigma_eps(
            W, Q2, cos_theta, phi, hep.ε_T(W, Q2, Eb), h)

    def interp_dsigma(self, *args, **kvargs):
        return np.ravel(self.interp_dsigma_v(*args, **kvargs))[0]

    def interp_dsigma_e_v(self, W, Q2, cos_theta, phi, Eb, h=1):
        """Electron scattering cross-section"""
        ε = hep.ε_T(W, Q2, Eb)
        return hep.Γ_ν(W, Q2, Eb, ε) * self.interp_dsigma_eps(
            W, Q2, cos_theta, phi, ε, h)

    def interp_dsigma_e(self, *args, **kvargs):
        return np.ravel(self.interp_dsigma_e_v(*args, **kvargs))[0]

    def dsigma_minmax(self, Eb, h=1):
        raise NotImplementedError(
            "Automatic calculation of maximum differential cross-section value"
            " not implemented yet")


class InterpSigmaLinearND(InterpSigma):
    def __init__(self, Amplitude, model, channel):
        super().__init__(Amplitude, model, channel)
        logger.debug('Interpolator initialization')
        self.interpolator = scipy.interpolate.LinearNDInterpolator(
            self.points, self.data)
        logger.debug('Interpolator initialized')

    def interp_H(self, w, q2, cos_theta):
        grid_w, grid_q2, grid_cθ = np.meshgrid(w, q2, cos_theta)
        return self.interpolator((grid_w, grid_q2, grid_cθ))


class InterpSigmaCached(InterpSigma):
    def __init__(self, model, channel):
        fname = f"cache/{model}_{channel}.npz"
        try:
            npz = np.load(fname)
            self.data = npz['amplitudes']
            self.points = npz['points']
            #logger.info(self.data)
            logger.info(f"Data loaded from file cache '{fname}'")
        except FileNotFoundError:
            from clasfw.models import Amplitude, Model, Channel
            from clasfw.app import create_app
            app = create_app()
            with app.test_request_context():
                o_model = Model.by_name(model)
                o_channel = Channel.by_name(channel)
                super().__init__(Amplitude, o_model, o_channel)
            np.savez(fname,
                amplitudes=self.data,
                points=self.points)
            logger.info(f"Data saved to cache file '{fname}'")
        logger.info('Interpolator initialization')
        self.interpolator = scipy.interpolate.LinearNDInterpolator(
            self.points, self.data)
        logger.info('Interpolator initialized')

    def interp_H(self, w, q2, cos_theta):
        grid_w, grid_q2, grid_cθ = np.meshgrid(w, q2, cos_theta)
        return self.interpolator((grid_w, grid_q2, grid_cθ))


import pickle
# amplitudes multiplied to special correction factor
class InterpSigmaCorrectedCached(InterpSigma):
    def __init__(self, model, channel):
        fname = f"cache/{model}_{channel}_prepared_interpolator_cache.pickle"
        try:
            with open(fname, 'rb') as fh:
                logger.info(f"Loading data from file cache '{fname}'")
                self.interpolator = pickle.load(fh)
            logger.info(f"Data loaded")
        except FileNotFoundError:
            from clasfw.models import Amplitude, Model, Channel
            from clasfw.app import create_app
            app = create_app()
            with app.test_request_context():
                o_model = Model.by_name(model)
                o_channel = Channel.by_name(channel)
                super().__init__(Amplitude, o_model, o_channel)

            self.correct_maid_H56()

            logger.info('Interpolator initialization')
            self.interpolator = scipy.interpolate.LinearNDInterpolator(
                self.points, self.data)
            logger.info('Interpolator initialized')
            with open(fname, 'wb') as fh:
                pickle.dump(self.interpolator, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Data saved to cache file '{fname}'")

    def interp_H(self, w, q2, cos_theta):
        grid_w, grid_q2, grid_cθ = np.meshgrid(w, q2, cos_theta)
        return self.interpolator((grid_w, grid_q2, grid_cθ))

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

    from argparse import ArgumentParser
    parser = ArgumentParser(
        fromfile_prefix_chars='@',
        description=
            'Interpolated differential cross-section 3D plot'
            ' from MAID helicity amplitudes data')
    parser.add_argument('-W', type=float, default=1.5,
        help='Final state invariant mass W, GeV')
    parser.add_argument('-Q2', type=float, default=1.0,
        help='Photon virtuality Q^2, GeV^2')
    parser.add_argument('--ebeam', '-E', type=float, default=10.6,
        help='Beam energy E, GeV')
    parser.add_argument('--helicity', '-H', type=int, default=1,
        choices=[-1, 1],
        help='Electron helicity')
    parser.add_argument('--no-flux', action="store_true",
        help='Do not multiply to virtual photon flux')
    parser.add_argument('--channel', '-C', type=str, default='pi0 p',
        choices=['pi+ n', 'pi0 p', 'pi- p', 'pi0 n'],
        help='Channel')
    args = parser.parse_args()

    W  = args.W
    Q2 = args.Q2
    E_beam = args.ebeam
    channel_name = args.channel

    logger.info('Loading data')

    amplitudes = InterpSigmaCorrectedCached('maid', channel_name)

    if 0:
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
        R_interp = amplitudes.interp_R(W, Q2, cos_theta)
        print(R_interp)
        print(R == R_interp)

    #cos_theta = np.linspace(-1, 1, 10)
    ε = 0.0000001
    cos_theta = np.arange(-1, 1 +ε, 0.01)
    #cos_theta = 0
    #cos_theta = 1
    grid_R = amplitudes.interp_R(W, Q2, cos_theta )
    logger.debug(f'SHAPE OF R: {grid_R.shape}')
    logger.debug(f'R: {grid_R}')

    qu_index = 0
    grid_R = grid_R[:,:,:,qu_index]

    logger.debug(f'SHAPE OF R: {grid_R.shape}')
    logger.debug(f'R: {grid_R}')

    phi = np.deg2rad(np.linspace(0, 360, 120+1))

    if args.no_flux:
        grid_dsig = amplitudes.interp_dsigma_v(
            W, Q2, cos_theta, phi, E_beam, h=args.helicity)
    else:
        grid_dsig = amplitudes.interp_dsigma_e_v(
            W, Q2, cos_theta, phi, E_beam, h=args.helicity)
    logger.debug(grid_dsig.shape)

    import plotly.graph_objects as go
    fig = go.Figure(
        layout_scene_xaxis_title='φ, rad',
        layout_scene_yaxis_title='cos θ',
        layout_scene_zaxis_title='dσ/dΩ, µb',
        layout_title=
            f'{channel_name}: Q² = {Q2} GeV², W = {W} GeV',
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
