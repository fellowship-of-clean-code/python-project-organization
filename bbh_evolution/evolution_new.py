from __future__ import annotations
from abc import ABC, abstractmethod
import warnings
from typing import Tuple, Union, Type, Optional
import numpy as np
import constants as const

from numba import njit


class Env(ABC):
    """Generic environment for a BBH to be in."""

    __slots__ = ()

    def __init__(self, binary: Optional[CompactBinary] = None):
        self._binary = binary

    @property
    def binary(self):
        return self._binary

    @binary.setter
    def binary(self, new_binary):
        self._binary = new_binary

    def __eq__(self, other):
        """Two environments are "equal"
        (i.e. satisfying env1 == env2)
        if they are of the same type (Vacuum, Cluster...).
        """

        return isinstance(other, self.__class__)

    @abstractmethod
    def evolution_derivatives(self) -> Tuple[float, float]:
        """Abstract method yielding the derivatives of the
        semimajor axis and the eccentricity for the binary.

        Returns:
            Tuple[float, float]: (a_dot, e_dot) (cgs units)
        """
        pass

    @abstractmethod
    def update(self) -> Env:
        """Update to the environment to be performed
        at each step in the evolution of the binary.

        Returns the environment itself, or possibly a different environment
        if the conditions require it (e. g. moving from a ClusterEnv to
        VacuumEnv if the binary is ejected from the cluster)

        While this is an abstract method, it also contains an implementation
        so that subclasses can explicitly refer back to it instead of having repetition
        (see, for example, VacuumEnv.update).
        """

        return self

    def should_stop_integration(self) -> bool:
        """The integration should be stopped in certain cases;
        it is stopped when this method returns True.

        While this is an abstract method, it also contains an implementation
        so that subclasses can explicitly refer back to it instead of having repetition
        (see, for example, VacuumEnv.should_stop_integration).

        Returns:
            (bool): whether the integration should be stopped.
        """

        return self._binary.t > const.tHubble * const.Myr


class VacuumEnv(Env):
    """Vacuum environment for a compact binary,
    which will evolve according only to GW emission.
    """

    __slots__ = ("_binary",)

    def __str__(self) -> str:
        return "Vacuum environment"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def evolution_derivatives(self) -> Tuple[float, float]:
        """Evolution equation for semimajor axis and eccentricity:
        peters 1964 formulas for GW-emission induced orbital decay only.

        Returns:
            a_dot (float): time derivative of a
            e_dot (float): time derivative of e
        """

        return evolution_derivatives_gw(
            self._binary.m1, self._binary.m2, self._binary.a, self._binary.e
        )

    def update(self) -> Env:
        return super().update()


class ClusterEnv(Env):

    __slots__ = (
        "_binary",
        "rho",
        "xi",
        "ki",
        "cluster_mass",
        "binary_fraction",
        "deviation_from_equipartition",
        "cluster_lifetime",
        "average_mass",
        "escape_velocity",
        "initial_escape_velocity",
        "initial_half_mass_relaxation_time",
    )

    def __init__(
        self,
        rho,
        xi,
        ki,
        cluster_mass,
        binary_fraction,
        deviation_from_equipartition,
        cluster_lifetime,
        average_mass,
        binary=None,
    ):
        """Initialize a star cluster.

        Args:
            rho (float): Average density of the cluster
            xi (float): (dimensionless)
            ki (float): (dimensionless)
            cluster_mass (float): (cgs units)
            binary_fraction (float): (dimensionless)
            deviation_from_equipartition (float): (dimensionless)
            cluster_lifetime (float): (cgs units)
            average_mass (float): (cgs units)
            binary (CompactBinary, optional): reference binary
            #TODO
        """

        super().__init__(binary)

        self.xi = xi
        self.ki = ki
        self.rho = rho
        self.cluster_mass = cluster_mass
        self.binary_fraction = binary_fraction
        # parameter from Spitzer 1969 (http://adsabs.harvard.edu/abs/1969ApJ...158L.139S)
        self.deviation_from_equipartition = deviation_from_equipartition
        self.cluster_lifetime = cluster_lifetime
        self.average_mass = average_mass

        self.initial_escape_velocity = self.compute_escape_velocity()
        self.escape_velocity = self.initial_escape_velocity

        self.initial_half_mass_relaxation_time = self.half_mass_relaxation_time

    def __str__(self) -> str:
        return "Cluster enviroment"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"rho={self.rho}, "
            f"xi={self.xi}, "
            f"ki={self.ki}, "
            f"cluster_mass={self.cluster_mass}, "
            f"binary_fraction={self.binary_fraction}, "
            f"deviation_from_equipartition={self.deviation_from_equipartition}, "
            f"cluster_lifetime={self.cluster_lifetime}, "
            f"average_mass={self.average_mass}"
            f")"
        )

    def compute_escape_velocity(self) -> float:
        return (
            40.0e5
            * (self.cluster_mass_solar / 1e5) ** (1.0 / 3.0)
            * (self.rho_msun_per_pc_cubed / 1e5) ** (1.0 / 6.0)
        )

    @property
    def sigma(self) -> float:
        """Velocity dispersion for the cluster."""
        return self.escape_velocity / 2

    @property
    def rho_msun_per_pc_cubed(self) -> float:
        return self.rho / const.msun * const.parsec**3.0

    @property
    def rhoc(self) -> float:
        """Core density of the cluster."""
        return const.c2hm_dens * self.rho

    @property
    def sigma1D(self) -> float:
        return self.sigma / np.sqrt(3)

    def half_mass_radius(self) -> float:

        rh = (3.0 * self.cluster_mass / (8.0 * np.pi * self.rho)) ** (1.0 / 3.0)
        if self._binary is None:
            return rh
        else:
            return rh * (
                0.15 * self._binary.t / self.initial_half_mass_relaxation_time + 1.0
            ) ** (2.0 / 3.0)

    @property
    def cluster_mass_solar(self) -> float:
        return self.cluster_mass / const.msun

    @property
    def half_mass_relaxation_time_Myr(self) -> float:

        return (
            7.5
            * (self.cluster_mass_solar / 1e5)
            / (self.rho_msun_per_pc_cubed / 1e5) ** 0.5
        )

    @property
    def half_mass_relaxation_time(self) -> float:
        return self.half_mass_relaxation_time_Myr * const.Myr

    @property
    def maximum_bh_mass(self) -> float:
        return 1e-3 * self.cluster_mass

    def time_three_body(self) -> float:
        """Timescale for the formation of a binary by three-body encounters,
        see Lee 1995, https://ui.adsabs.harvard.edu/abs/1995MNRAS.272..605L
        and also Ivanova et al 2005, https://ui.adsabs.harvard.edu/abs/2005MNRAS.358..572I
        """

        return (
            125.0
            * (1e6 / self.rho) ** 2
            * (self.sigma1D / 30.0e5 / self.deviation_from_equipartition) ** 9
            * (20.0 / self._binary.m1) ** 5
        )

    def a_ejection(self) -> float:
        """Semi-major axis below which binary is ejected by three-body encounters"""

        return (
            self.xi
            * self.average_mass
            * self.average_mass
            / (self._binary.m1 + self._binary.m2) ** 3.0
            * const.G
            * self._binary.m1
            * self._binary.m2
            / self.escape_velocity
            / self.escape_velocity
        )

    def a_gw_domination(self) -> float:
        """Semi-major axis below which GWs dominate the evolution; obtained assuming adot_ej=adot_gw (Baibhav et al. 2020)"""

        return (
            32.0
            * const.G**2.0
            / (5.0 * np.pi * self.xi * const.c**5.0)
            * (self.sigma / self.rho)
            * self._binary.m1
            * self._binary.m2
            * (self._binary.m1 + self._binary.m2)
            / ((1 - self._binary.e * self._binary.e) ** 3.5)
            * (
                1.0
                + 73.0 / 24.0 * self._binary.e * self._binary.e
                + 37.0 / 96.0 * self._binary.e**4
            )
        ) ** (1.0 / 5.0)

    def update(self) -> Env:

        self.escape_velocity = self.initial_escape_velocity * (
            0.15 * self._binary.t / self.initial_half_mass_relaxation_time + 1.0
        ) ** (-1.0 / 3.0)

        self.rho = (
            3.0 * self.cluster_mass / (8.0 * np.pi * self.half_mass_radius() ** 3)
        )

        if self.should_change_env():
            return VacuumEnv(self._binary)
        else:
            return self

    def should_change_env(self) -> bool:

        a_gw = self.a_gw_domination()
        a_ej = self.a_ejection()

        return not (
            (self._binary.t + self.time_three_body() <= self.cluster_lifetime)
            and (a_gw >= a_ej or self._binary.a >= a_ej > a_gw)
        )

    def evolution_derivatives(self) -> Tuple[float, float]:
        """Evolution equation for semimajor axis and eccentricity:
        peters 1964 formulas for GW-emission induced orbital decay
        plus hardening [REF]

        Returns:
            a_dot (float): time derivative of a
            e_dot (float): time derivative of e
        """

        return self._evolution_derivatives(
            self._binary.m1,
            self._binary.m2,
            self._binary.a,
            self._binary.e,
            self.sigma,
            self.rhoc,
            self.xi,
            self.ki,
        )

    @staticmethod
    @njit
    def _evolution_derivatives(m1, m2, a, e, sigma, rhoc, xi, ki):

        a_dot_gw, e_dot_gw = evolution_derivatives_gw(m1, m2, a, e)

        a_dot = a_dot_gw + -2.0 * np.pi * xi * const.G * rhoc / sigma * a**2

        e_dot = e_dot_gw + +ki * 2.0 * np.pi * xi * const.G * rhoc / sigma * a

        return (a_dot, e_dot)


class AGNDiskEnv(Env):

    __slots__ = (
        "_binary",
        "alpha",
        "cs",
        "sigmag",
    )

    def __init__(self, alpha, cs, sigmag, binary=None):
        """Initialize an AGN disk.

        Args:
            alpha (float): alpha-prescription for the disk viscosity, between 0 and 1
            cs (float): sound speed in the disk
            sigmag (float): surface density of the disk
        """

        super().__init__(binary)

        self.alpha = alpha
        self.cs = cs
        self.sigmag = sigmag

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"alpha={self.alpha}, "
            f"cs={self.cs}, "
            f"sigmag={self.sigmag})"
        )

    def __str__(self) -> str:
        return "AGN Disk environment"

    @classmethod
    def from_smbh_parameters(cls, alpha, m_smbh, r_from_smbh, h, f_g) -> Env:
        """Example of classmethod:
        initialize the same class with different parameters,
        by going back to the default __init__ method.
        """

        # best fit for sigma, from Ishibashi & Gröbner 2020
        sigma = 0.000871036 * const.c * (m_smbh / (1.0e9 * const.msun)) ** 0.228

        sigmag = f_g * sigma**2 / np.pi / const.G / r_from_smbh

        cs = h * np.sqrt(const.G * m_smbh / r_from_smbh)

        return cls(alpha, cs, sigmag)

    def evolution_derivatives(self) -> Tuple[float, float]:
        """Evolution equation for semimajor axis and eccentricity:
        peters 1964 formulas for GW-emission induced orbital decay,
        plus evolution equations in AGN disks.
        Ref: Ishibashi & Gröbner 2020, http://arxiv.org/abs/2006.07407


        Returns:
            a_dot (float): time derivative of a
            e_dot (float): time derivative of e
        """

        return self._evolution_derivatives(
            self._binary.m1,
            self._binary.m2,
            self._binary.a,
            self._binary.e,
            self.alpha,
            self.cs,
            self.sigmag,
        )

    @staticmethod
    @njit
    def _evolution_derivatives(m1, m2, a, e, alpha, cs, sigmag):

        a_dot_gw, e_dot_gw = evolution_derivatives_gw(m1, m2, a, e)
        a_dot = a_dot_gw + -24.0 * np.pi * alpha * cs * cs * sigmag * np.sqrt(
            m1 + m2
        ) * (1 + e) * (1 + e) * a**2.5 / (m1 * m2 * np.sqrt(const.G))

        e_dot = e_dot_gw + +12.0 * np.pi * alpha * cs * cs * sigmag * np.sqrt(
            m1 + m2
        ) * (1 + e) * (1 + e) * np.sqrt(1 - e * e) * a**1.5 / (
            m1 * m2 * np.sqrt(const.G)
        ) * (
            0.5 * e + 0.125 * e**3 + 0.0625 * e**5 + 0.0390625 * e**7
        )
        # this last line,
        # * (.5 * e + .125 * e**3 + .0625 * e**5 + 0.0390625 * e**7),
        #  is a Taylor expansion of
        # * (1-np.sqrt(1-e*e)) / e
        # it does not diverge for small e
        # and it has better numeric accuracy

        return (a_dot, e_dot)

    def update(self) -> Env:
        return super().update()


class CompactBinary:
    """Generic compact binary system.
    Implements evolution according to an adaptive-step Euler scheme.
    The evolutionary derivatives are defined by the environment.

    Attributes
    -----------
    m1 (float): mass of the primary (cgs units)
    m2 (float): mass of the secondary (cgs units)
    a (float): semimajor axis of the binary (cgs units)
    e (float): eccentricity of the binary, between 0 and 1
    env (Env): environment the binary will evolve in, which determines the
        ODE describing the update of a and e

    r_isco (float): ISCO of the binary (=6GM) (cgs units)
    r_threshold (float): a radius slightly above the ISCO,
        at which we stop the evolution of the binary (cgs units)
    timestep (float): timestep for the adaptive Euler scheme (cgs units),
        it is varied during the evolution
    t (float): time from the start of integration (cgs units)

    Class attributes
    ----------------
    tol_low (float, optional): if the relative variation of the
        semimajor axis falls below this value,
        raise the timestep by raise_timestep_by. Defaults to 1e-3.

    raise_timestep_by (float, optional): defaults to 2.

    tol_high (float, optional): if the relative variation of the
        semimajor axis rises above this value,
        lower the timestep by lower_timestep_by. Defaults to 1e-2.

    lower_timestep_by (float, optional): defaults to 10.


    Methods
    --------
    evolution_step:
        Evolve the binary by a single timestep,
        changing the timestep if the relative variation
        of the semimajor axis becomes too large or too small.

    evolve:
        Evolve the binary up to merger,
        which is computed by checking that the
        semimajor axis is a small tolerance above the ISCO

    evolve_saving_iterations:
        the same thing as "evolve",
        but returning a BBHList object
        which can be used to check on the evolution
    """

    thr_above_isco = 1.1

    tol_low = 3e-6
    tol_high = 0.004
    raise_timestep_by = 4.0
    lower_timestep_by = 1.75

    __slots__ = (
        "_env",
        "m1",
        "m2",
        "timestep",
        "t",
        "a",
        "e",
        "r_isco",
        "r_threshold",
        "mtot",
    )

    def __init__(
        self,
        mass_1: float,
        mass_2: float,
        semimajor_axis: float,
        eccentricity: float,
        env: Env = VacuumEnv(),
    ):
        """Initialize a binary system.
        All attributes are internally kept in SI units,
        but they are given in other units (specified case by case).

        Args:
            mass_1 (float): mass of the primary, in solar masses
            mass_2 (float): mass of the secondary, in solar masses
            semimajor_axis (float): semimajor axis of the binary, in solar radii
            eccentricity (float): eccentricity of the binary, between 0 and 1
            env (Env): environment of the binary
        """

        self.e = eccentricity

        assert mass_1 >= mass_2

        # go cgs
        self.m1 = mass_1 * const.msun
        self.m2 = mass_2 * const.msun
        self.mtot = self.m1 + self.m2
        self.a = semimajor_axis * const.Rsun

        self.r_isco = 6.0 * const.G * self.mtot / const.c**2
        self.r_threshold = self.r_isco * self.thr_above_isco

        self.timestep = 0.1 * self.gw_evolution_timescale()

        self.t = 0.0

        env.binary = self

        self._env = env

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"mass_1={self.m1_sun_masses}, "
            f"mass_2={self.m2_sun_masses}, "
            f"semimajor_axis={self.semimajor_axis_sun_radii}, "
            f"eccentricity={self.e}, "
            f"env={self.env!r})"
        )

    @property
    def env(self) -> Env:
        """Environment in which to evolve the binary;
        it provides the evolutionary derivatives as well as the


        Returns:
            Env: environment object.
        """
        return self._env

    @env.setter
    def env(self, env) -> None:
        self._env = env

    def gw_evolution_timescale(self) -> float:
        return (
            5.0
            / 256.0
            * const.c5
            / const.G3
            * self.a**4
            * (1.0 - self.e * self.e) ** 3.5
            / self.m1
            / self.m2
            / (self.mtot)
        )

    @property
    def m1_sun_masses(self) -> float:
        return self.m1 / const.msun

    @property
    def m2_sun_masses(self) -> float:
        return self.m2 / const.msun

    @property
    def semimajor_axis_sun_radii(self) -> float:
        return self.a / const.Rsun

    @property
    def time_Myr(self) -> float:
        return self.t / const.Myr

    def evolution_derivatives(self) -> Tuple[float, float]:
        """Evolutionary derivatives for the binary's evolution.
        Takes no arguments, returns the time derivatives of the semimajor axis
        and the eccentricity.
        """
        return self._env.evolution_derivatives()

    def evolution_step(self) -> Tuple[float, float]:
        """Evolve the binary system by a single step
        using an adaptive-timestep Euler scheme.
        """

        anew, enew = self.euler_step(
            *self.evolution_derivatives(), self.timestep, self.a, self.e
        )

        relative_semiaxis_variation = abs(anew - self.a) / self.a

        # recursive implementation of the adaptive timestep
        # this should typically only go 1 level deep,
        # but the recursiveness makes it robust to wildly wrong
        # guesses for the initial timestep

        if relative_semiaxis_variation < self.tol_low:
            self.timestep *= self.raise_timestep_by
            self.evolution_step()

        elif relative_semiaxis_variation > self.tol_high:
            self.timestep /= self.lower_timestep_by
            self.evolution_step()

        else:
            self.a = anew
            self.e = enew

        self.t += self.timestep

    def combined_iteration_step(self) -> None:
        """Evolve both the binary and the environment by one step."""

        self.evolution_step()
        self._env = self._env.update()

        if self.e < 0:
            warnings.warn("Eccentricity is negative. Setting it to zero.")
            self.e = 0

        # (also asserting a > 0 is redundant,
        # since if it were < 0 the while loop would terminate)

    def should_stop_integration(self) -> bool:
        return self.merged or self._env.should_stop_integration()

    @property
    def merged(self) -> bool:
        return self.a < self.r_threshold

    def evolve(self) -> None:
        while not self.should_stop_integration():
            self.combined_iteration_step()

    @staticmethod
    @njit
    def euler_step(
        a_dot: float, e_dot: float, timestep: float, a: float, e: float
    ) -> Tuple[float, float]:
        return (a + a_dot * timestep, e + e_dot * timestep)


class BinaryBlackHole(CompactBinary):
    __slots__ = CompactBinary.__slots__


@njit
def evolution_derivatives_gw(
    m1: float, m2: float, a: float, e: float
) -> Tuple[float, float]:
    """Contribution to binary evolution
    from energy loss due to GW emission

    Args:
        m1 (float): primary mass (cgs units)
        m2 (float): secondary mass (cgs units)
        a (float): semimajor axis (cgs units)
        e (float): eccentricity (cgs units)

    Returns:
        (a_dot, e_dot) (Tuple[float, float]): semimajor axis and eccentricity derivatives
    """
    return (
        -64.0
        / 5.0
        * const.G3
        * m1
        * m2
        * (m1 + m2)
        / (const.c5 * a**3 * (1 - e**2) ** 3.5)
        * (1.0 + 73.0 / 24.0 * e**2 + 37.0 / 96.0 * e**4),
        -304.0
        / 15.0
        * e
        * const.G3
        * m1
        * m2
        * (m1 + m2)
        / (const.c5 * a**4 * (1 - e**2) ** 2.5)
        * (1.0 + 121.0 / 304.0 * e**2),
    )


if __name__ == "__main__":

    agn_env = AGNDiskEnv.from_smbh_parameters(
        alpha=0.1,
        m_smbh=1e7 * const.msun,
        r_from_smbh=0.1 * const.parsec,
        h=0.01,
        f_g=0.1,
    )

    cluster_env = ClusterEnv(
        rho=3e-19,
        xi=3.0,
        ki=0.1,
        cluster_mass=1.5e6 * const.msun,
        binary_fraction=0.01,
        deviation_from_equipartition=1.0,
        cluster_lifetime=const.tHubble * const.Myr,
        average_mass=const.msun,
    )

    vacuum_env = VacuumEnv()

    environments = {
        "Vacuum environment": vacuum_env,
        "AGN disk environment": agn_env,
        "Cluster environment": cluster_env,
    }

    for name, env in environments.items():
        bbh = BinaryBlackHole(
            mass_1=30.0, mass_2=30.0, semimajor_axis=100.0, eccentricity=0.7, env=env
        )

        print(name)
        print(f"Initial time: {bbh.time_Myr:.1f}Myr")
        bbh.evolve()
        print(f"Final time: {bbh.time_Myr:.1f}Myr")
        print(bbh)
        print(bbh.merged)
        print()
