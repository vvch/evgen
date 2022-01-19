#!/usr/bin/python3
import os, sys
import numpy as np
import scipy.interpolate
import logging
logger = logging.getLogger(__name__)

import hep
import hep.amplitudes


class InterpSigma:
    available_channels = ['pi+ n', 'pi0 p', 'pi- p', 'pi0 n']

    def __init__(self, Amplitude, model, channel):
        self.load_from_db()
        self.init_interpolator()

    def load_from_db(self, Amplitude, model, channel):
        data = Amplitude.query.filter_by(
            model=model,
            channel=channel,
        ).order_by(
            Amplitude.q2,
            Amplitude.w,
            Amplitude.cos_theta,
        ).with_entities(
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

        logger.info('Data loaded from the DB')
        logger.debug(f'SHAPE OF POINTS: {self.points.shape}')
        logger.debug(f'POINTS:\n{self.points}')
        logger.debug(f'SHAPE OF DATA: {self.data.shape}')
        logger.debug(f'DATA:\n{self.data}')

    def load_from_db_by_names(self, model, channel):
        from clasfw.models import Amplitude, Model, Channel
        from clasfw.app import create_app
        app = create_app()
        with app.test_request_context():
            self.load_from_db(Amplitude,
                Model.by_name(model),
                Channel.by_name(channel))

    def init_interpolator(self):
        logger.debug('Interpolator initialization')
        self.interpolator = scipy.interpolate.LinearNDInterpolator(
            self.points, self.data)
        logger.debug('Interpolator initialized')

    def correct_maid_H56(self):
        logger.info('Correcting MAID data')
        for i in range(len(self.points)):
            W, Q2 = self.points[i, 0:2]
            self.data[i, 4:6] *= hep.amplitudes.H56_maid_correction_factor(W, Q2)  ##  H5, H6

    def interp_H(self, w, q2, cos_theta):
        grid_w, grid_q2, grid_cθ = np.meshgrid(w, q2, cos_theta)
        return self.interpolator((grid_w, grid_q2, grid_cθ))

    def interp_R(self, w, q2, cos_theta):
        grid_R = self.interp_H(w, q2, cos_theta)
        grid_R = np.apply_along_axis(
            hep.amplitudes.ampl_to_R, 3, grid_R,  #  3rd axis of grid_R with amplitudes
            # np.sum, 3, grid_R,
        )
        #grid_R = grid_R[:,:,:,self.qu_index]
        return grid_R

    def interp_dsigma_comps(self, w, q2, cos_theta):
        return self.interp_R(w, q2, cos_theta) * hep.amplitudes.R_to_dsigma_factor(w, q2)

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
        """Electron scattering cross-section"""
        return np.ravel(self.interp_dsigma_e_v(*args, **kvargs))[0]

    def dsigma_minmax(self, Eb, h=1):
        raise NotImplementedError(
            "Automatic calculation of maximum differential cross-section value"
            " not implemented yet")


class InterpSigmaCached(InterpSigma):
    def __init__(self, model, channel):
        fname = os.path.join(os.path.dirname(__file__), 'cache',
            f"{model}_{channel}.npz")
        try:
            npz = np.load(fname)
            self.data = npz['amplitudes']
            self.points = npz['points']
            #logger.info(self.data)
            logger.info(f"Data loaded from file cache '{fname}'")
        except FileNotFoundError:
            self.load_from_db_by_names(model, channel)
            np.savez(fname,
                amplitudes=self.data,
                points=self.points)
            logger.info(f"Data saved to cache file '{fname}'")
        self.init_interpolator()


import pickle
# amplitudes multiplied to special correction factor
class InterpSigmaCorrectedCached(InterpSigma):
    def __init__(self, model, channel):
        fname = os.path.join(os.path.dirname(__file__), 'cache',
            f"{model}_{channel}_prepared_interpolator_cache.pickle")
        try:
            with open(fname, 'rb') as fh:
                logger.info(f"Loading data from file cache '{fname}'")
                self.interpolator = pickle.load(fh)
            logger.info(f"Data loaded")
        except FileNotFoundError:
            self.load_from_db_by_names(model, channel)
            self.correct_maid_H56()
            self.init_interpolator()
            with open(fname, 'wb') as fh:
                pickle.dump(self.interpolator, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Data saved to cache file '{fname}'")


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
    parser.add_argument('--no-plot', action="store_true",
        help='Skip plot')
    parser.add_argument('--channel', '-C', type=str,
        default=InterpSigmaCorrectedCached.available_channels[0],
        choices=InterpSigmaCorrectedCached.available_channels,
        help='Channel')
    parser.add_argument('--output', '-o', type=str,
        help='Output html file name (show in browser if empty)')
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
    #cos_theta = np.arange(-1, 1 +ε, 0.01)
    cos_theta = np.arange(-1, 1 +ε, 0.1)
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

    grid_dsig = (
        amplitudes.interp_dsigma_v
            if args.no_flux else
        amplitudes.interp_dsigma_e_v
    )(W, Q2, cos_theta, phi, E_beam, h=args.helicity)

    logger.debug(grid_dsig.shape)
    DCS = grid_dsig[0,0]
    DCS_min = DCS.min()
    DCS_max = DCS.max()
    print(f"DCS min: {DCS_min:g}, \tmax: {DCS_max}")

    if args.no_plot:
        sys.exit()

    import plotly.graph_objects as go
    fig = go.Figure(
        layout_scene_xaxis_title='φ, rad',
        layout_scene_yaxis_title='cos θ',
        layout_scene_zaxis_title='dσ/dΩ, µb',
        layout_title=
            f'{channel_name}: Q² = {Q2} GeV², W = {W} GeV\n'
            f' min={DCS_min:g} max={DCS_max:g}',
    )
    fig.add_surface(
        x=phi,
        y=cos_theta,
        z=DCS,
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
    if args.output:
        fig.write_html(args.output)
    else:
        fig.show()
    logger.info('Done')
