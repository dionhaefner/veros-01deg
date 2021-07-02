__VEROS_VERSION__ = "1.3.3"

import os
import h5netcdf

from veros import runtime_state as rst, runtime_settings as rs

os.environ["MPI4JAX_USE_CUDA_MPI"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = str(rst.proc_rank)

rs.backend = "jax"
rs.device = "gpu"
rs.float_type = "float32"
rs.petsc_options = (
    "-ksp_type gmres -ksp_rtol 1e-8 "
    "-pc_gamg_agg_nsmooths 1 -pc_gamg_process_eq_limit 100000"
)
rs.monitor_streamfunction_residual = False

from veros import VerosSetup, tools, time, veros_routine, veros_kernel, KernelOutput, logger
from veros.variables import Variable
from veros.distributed import get_chunk_slices
from veros.core.operators import numpy as npx, update, at


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = tools.get_assets(
    "global_01deg", os.path.join(BASE_PATH, "assets.json"), skip_md5=True
)


class GlobalEddyResolvingSetup(VerosSetup):
    """Global 0.1 degree model with either 60 or 120 vertical levels."""

    num_layers = 60  # choose 60 or 120

    @veros_routine
    def set_parameter(self, state):
        """
        set main parameters
        """
        settings = state.settings

        settings.identifier = f"01deg-{self.num_layers}layers"

        settings.nx = 3600
        settings.ny = 1600
        settings.nz = self.num_layers

        settings.dt_mom = 120.0
        settings.dt_tracer = 120.0
        settings.runlen = 360 * 86400

        settings.restart_frequency = 30 * 86400

        settings.x_origin = 90.0
        settings.y_origin = -80.0

        settings.coord_degree = True
        settings.enable_cyclic_x = True

        settings.enable_biharmonic_mixing = True
        settings.K_hbi = 1e9

        settings.enable_biharmonic_friction = True
        settings.A_hbi = 1e11
        settings.biharmonic_friction_cosPower = 3

        settings.enable_tempsalt_sources = True
        settings.enable_implicit_vert_friction = True

        settings.eq_of_state_type = 5

        # isoneutral
        settings.enable_neutral_diffusion = True
        settings.enable_skew_diffusion = True
        settings.K_iso_0 = 1000.0
        settings.K_iso_steep = 200.0
        settings.iso_dslope = 0.001
        settings.iso_slopec = 0.004

        # tke
        settings.enable_tke = True
        settings.c_k = 0.1
        settings.c_eps = 0.7
        settings.alpha_tke = 30.0
        settings.mxl_min = 1e-8
        settings.tke_mxl_choice = 2
        settings.kappaM_min = 2e-4
        settings.kappaH_min = 2e-5
        settings.enable_kappaH_profile = True
        settings.enable_tke_superbee_advection = True

        # eke
        settings.enable_eke = False
        settings.K_gm_0 = 100.0

        settings.enable_idemix = False

        # custom variables
        state.dimensions["nmonths"] = 12
        state.var_meta.update(
            t_star=Variable(
                "t_star", ("xt", "yt", "nmonths"), "", "", time_dependent=False
            ),
            s_star=Variable(
                "s_star", ("xt", "yt", "nmonths"), "", "", time_dependent=False
            ),
            qnec=Variable(
                "qnec", ("xt", "yt", "nmonths"), "", "", time_dependent=False
            ),
            qnet=Variable(
                "qnet", ("xt", "yt", "nmonths"), "", "", time_dependent=False
            ),
            qsol=Variable(
                "qsol", ("xt", "yt", "nmonths"), "", "", time_dependent=False
            ),
            divpen_shortwave=Variable(
                "divpen_shortwave", ("zt",), "", "", time_dependent=False
            ),
            taux=Variable(
                "taux", ("xt", "yt", "nmonths"), "", "", time_dependent=False
            ),
            tauy=Variable(
                "tauy", ("xt", "yt", "nmonths"), "", "", time_dependent=False
            ),
        )

    def _get_data(self, var, idx=None):
        if idx is None:
            idx = Ellipsis
        elif not isinstance(idx, slice):
            idx = idx[::-1]

        kwargs = {}
        if rst.proc_num > 1:
            kwargs.update(
                driver="mpio",
                comm=rs.mpi_comm,
            )

        forcing_file = DATA_FILES[f"forcing-{self.num_layers}layers"]

        with h5netcdf.File(forcing_file, "r", **kwargs) as f:
            var_obj = f.variables[var]
            return npx.array(var_obj[idx]).T

    @veros_routine
    def set_grid(self, state):
        vs = state.variables
        settings = state.settings

        x_idx, _ = get_chunk_slices(settings.nx, settings.ny, ("xt",))
        vs.dxt = update(vs.dxt, at[2:-2], self._get_data("dxt", x_idx[0]))

        y_idx, _ = get_chunk_slices(settings.nx, settings.ny, ("yt",))
        vs.dyt = update(vs.dyt, at[2:-2], self._get_data("dyt", y_idx[0]))

        vs.dzt = self._get_data("dzt")[::-1]

    @veros_routine
    def set_coriolis(self, state):
        vs = state.variables
        settings = state.settings
        vs.coriolis_t = update(
            vs.coriolis_t,
            at[...],
            2 * settings.omega * npx.sin(vs.yt[npx.newaxis, :] / 180.0 * settings.pi),
        )

    @veros_routine
    def set_topography(self, state):
        vs = state.variables
        settings = state.settings
        idx, _ = get_chunk_slices(settings.nx, settings.ny, ("xt", "yt"))
        vs.kbot = update(vs.kbot, at[2:-2, 2:-2], self._get_data("kbot", idx=idx))

    @veros_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        rpart_shortwave = 0.58
        efold1_shortwave = 0.35
        efold2_shortwave = 23.0

        idx, _ = get_chunk_slices(settings.nx, settings.ny, ("xt", "yt"))
        idx += (Ellipsis,)

        # initial conditions
        temp_data = self._get_data("temperature", idx)
        vs.temp = update(
            vs.temp,
            at[2:-2, 2:-2, :, 0],
            temp_data[..., ::-1] * vs.maskT[2:-2, 2:-2, :],
        )
        vs.temp = update(
            vs.temp,
            at[2:-2, 2:-2, :, 1],
            temp_data[..., ::-1] * vs.maskT[2:-2, 2:-2, :],
        )

        salt_data = self._get_data("salinity", idx)
        vs.salt = update(
            vs.salt,
            at[2:-2, 2:-2, :, 0],
            salt_data[..., ::-1] * vs.maskT[2:-2, 2:-2, :],
        )
        vs.salt = update(
            vs.salt,
            at[2:-2, 2:-2, :, 1],
            salt_data[..., ::-1] * vs.maskT[2:-2, 2:-2, :],
        )

        # wind stress on MIT grid
        vs.taux = update(vs.taux, at[2:-2, 2:-2, :], self._get_data("tau_x", idx))
        vs.tauy = update(vs.tauy, at[2:-2, 2:-2, :], self._get_data("tau_y", idx))

        qnec_data = self._get_data("dqdt", idx)
        vs.qnec = update(
            vs.qnec,
            at[2:-2, 2:-2, :],
            qnec_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis],
        )

        qsol_data = self._get_data("swf", idx)
        vs.qsol = update(
            vs.qsol,
            at[2:-2, 2:-2, :],
            -qsol_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis],
        )

        # SST and SSS
        sst_data = self._get_data("sst", idx)
        vs.t_star = update(
            vs.t_star,
            at[2:-2, 2:-2, :],
            sst_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis],
        )

        sss_data = self._get_data("sss", idx)
        vs.s_star = update(
            vs.s_star,
            at[2:-2, 2:-2, :],
            sss_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis],
        )

        """
        Initialize penetration profile for solar radiation and store divergence in divpen
        note that pen is set to 0.0 at the surface instead of 1.0 to compensate for the
        shortwave part of the total surface flux
        """
        swarg1 = vs.zw / efold1_shortwave
        swarg2 = vs.zw / efold2_shortwave
        pen = rpart_shortwave * npx.exp(swarg1) + (1.0 - rpart_shortwave) * npx.exp(
            swarg2
        )
        pen = update(pen, at[-1], 0.0)

        vs.divpen_shortwave = update(
            vs.divpen_shortwave, at[1:], (pen[1:] - pen[:-1]) / vs.dzt[1:]
        )
        vs.divpen_shortwave = update(vs.divpen_shortwave, at[0], pen[0] / vs.dzt[0])

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        vs.update(set_forcing_kernel(state))

    @veros_routine
    def set_diagnostics(self, state):
        # frequent CFL reports
        state.diagnostics["cfl_monitor"].output_frequency = (
            10 * state.settings.dt_tracer
        )

        # bi-monthly snapshots
        state.diagnostics["snapshot"].output_frequency = 360 * 86400 / 24.0
        state.diagnostics["snapshot"].output_variables = [
            "dxt",
            "dxu",
            "dyt",
            "dyu",
            "dzt",
            "dzw",
            "ht",
            "temp",
            "salt",
            "prho",
            "u",
            "v",
            "w",
            "forc_temp_surface",
            "forc_salt_surface",
            "forc_rho_surface",
            "surface_taux",
            "surface_tauy",
            "psi",
        ]

        # monthly means
        state.diagnostics["averages"].output_frequency = 360 * 86400 / 12.0
        state.diagnostics["averages"].sampling_frequency = state.settings.dt_tracer
        state.diagnostics["averages"].output_variables = [
            "temp",
            "salt",
            "u",
            "v",
            "w",
            "surface_taux",
            "surface_tauy",
            "psi",
        ]

    @veros_routine
    def after_timestep(self, state):
        vs = state.variables
        settings = state.settings

        if settings.dt_tracer < 150 and vs.time > 90 * 86400:
            logger.info("Setting time step to 150s")
            with settings.unlock():
                settings.dt_tracer = 150
                settings.dt_mom = 150

        elif settings.dt_tracer < 180 and vs.time > 180 * 86400:
            logger.info("Setting time step to 180s")
            with settings.unlock():
                settings.dt_tracer = 180
                settings.dt_mom = 180


@veros_kernel
def set_forcing_kernel(state):
    vs = state.variables
    settings = state.settings

    t_rest = 30.0 * 86400.0
    cp_0 = 3991.86795711963  # J/kg /K

    year_in_seconds = time.convert_time(1.0, "years", "seconds")
    (n1, f1), (n2, f2) = tools.get_periodic_interval(
        vs.time, year_in_seconds, year_in_seconds / 12.0, 12
    )

    # linearly interpolate wind stress and shift from MITgcm U/V grid to this grid
    vs.surface_taux = update(
        vs.surface_taux, at[:-1, :], f1 * vs.taux[1:, :, n1] + f2 * vs.taux[1:, :, n2]
    )
    vs.surface_tauy = update(
        vs.surface_tauy, at[:, :-1], f1 * vs.tauy[:, 1:, n1] + f2 * vs.tauy[:, 1:, n2]
    )

    if settings.enable_tke:
        vs.forc_tke_surface = update(
            vs.forc_tke_surface,
            at[1:-1, 1:-1],
            npx.sqrt(
                (
                    0.5
                    * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1])
                    / settings.rho_0
                )
                ** 2
                + (
                    0.5
                    * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2])
                    / settings.rho_0
                )
                ** 2
            )
            ** (3.0 / 2.0),
        )

    # W/m^2 K kg/J m^3/kg = K m/s
    t_star_cur = f1 * vs.t_star[..., n1] + f2 * vs.t_star[..., n2]
    qqnec = f1 * vs.qnec[..., n1] + f2 * vs.qnec[..., n2]
    qqnet = f1 * vs.qnet[..., n1] + f2 * vs.qnet[..., n2]
    vs.forc_temp_surface = (
        (qqnet + qqnec * (t_star_cur - vs.temp[..., -1, vs.tau]))
        * vs.maskT[..., -1]
        / cp_0
        / settings.rho_0
    )
    s_star_cur = f1 * vs.s_star[..., n1] + f2 * vs.s_star[..., n2]
    vs.forc_salt_surface = (
        1.0
        / t_rest
        * (s_star_cur - vs.salt[..., -1, vs.tau])
        * vs.maskT[..., -1]
        * vs.dzt[-1]
    )

    # apply simple ice mask
    mask1 = vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] > -1.8
    mask2 = vs.forc_temp_surface > 0
    ice = npx.logical_or(mask1, mask2)
    vs.forc_temp_surface *= ice
    vs.forc_salt_surface *= ice

    # solar radiation
    if settings.enable_tempsalt_sources:
        vs.temp_source = (
            (f1 * vs.qsol[..., n1, None] + f2 * vs.qsol[..., n2, None])
            * vs.divpen_shortwave[None, None, :]
            * ice[..., None]
            * vs.maskT[..., :]
            / cp_0
            / settings.rho_0
        )

    return KernelOutput(
        surface_taux=vs.surface_taux,
        surface_tauy=vs.surface_tauy,
        temp_source=vs.temp_source,
        forc_tke_surface=vs.forc_tke_surface,
        forc_temp_surface=vs.forc_temp_surface,
        forc_salt_surface=vs.forc_salt_surface,
    )
