# Object-oriented BBH evolution code

## Context

This code is based on the evolution equations implemented in the
[FASTCLUSTER](https://gitlab.com/micmap/fastcluster_open) code, by Michela Mapelli.

It's evolving black hole binaries in different environments - 
a binary will shrink by emitting gravitational waves, as well as due to
effects in its environment.
For example, in a dense star cluster there will be stochastic effects due to 
a third body affecting the binary; or in an AGN disk there will be drag induced 
by the gas.
These can all be modelled as an ODE for the two binary parameters $a$, the semimajor axis
of the orbit, and $e$, its eccentricity.

The point of this code is to compare the evolution in these various environments.
Its original version is available in the linked repo and is reproduced here as `evolution_old.py`.

The functions evolving binaries in that code are `peters` and `peters_evol`, depending 
on whether the environment was vacuum or a cluster.
This entailed a large amount of code repetition.

## Object-oriented version

The code is able to evolve different compact binaries in different environments.
A lot of the functionality needed to accomplish this is the same across all of them, 
with only a small amount needing to be specified.

There is an abstract `Env` class with three implementations: `VacuumEnv`, `ClusterEnv`
and `AGNDiskEnv`.

The `Env` class defines an interface, a promise of what every subclass needs to be able to do:
every one needs to implement the methods `evolution_derivatives` and `update`.

Then, in the `CompactBinary` class we use a strategy known as __dependency injection__:
in the initialization the class takes an `Env` object, which can be any of the environments
above, and it uses only the "public-facing" methods of this object, the derivatives and the update.

This way, the `CompactBinary` is inter-operable across all possible environments -
it's even possible for the environment class to be changed during the evolution!

### Small nice things

#### `__repr__`

What happens when we print out an object? 

```python
>>> from evolution_new import *
>>> bbh = BinaryBlackHole(mass_1=30., mass_2=30., semimajor_axis=100., eccentricity=0.5, env=VacuumEnv())
>>> print(bbh)
BinaryBlackHole(mass_1=30.0, mass_2=30.0, semimajor_axis=100.0, eccentricity=0.5, env=VacuumEnv())
>>> bbh.evolve()
>>> print(bbh)
BinaryBlackHole(mass_1=30.0, mass_2=30.0, semimajor_axis=95.3108267367454, eccentricity=0.4823477512047497, env=VacuumEnv())
```

This is calling the `bbh.__repr__()` method, which is implemented in the class:
```python
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"mass_1={self.m1_sun_masses}, "
            f"mass_2={self.m2_sun_masses}, "
            f"semimajor_axis={self.semimajor_axis_sun_radii}, "
            f"eccentricity={self.e}, "
            f"env={self.env!r})"
        )
```

If this was not implemented, we'd get an ugly thing like 
`<evolution_new.BinaryBlackHole object at 0x7f146da00340>`.

#### `classmethod`s

The AGN disk environment can be initialized in different ways.
The parameters it needs are `alpha`, `cs`  and `sigmag` (disk viscosity prescription, speed of sound
and disk surface density); but we might want to compute these 
parameters from the ones of the supermassive black hole, `alpha, m_smbh, r_from_smbh, h, f_g`
(disk viscosity prescription, mass of the supermassive black hole, radius and height from the disk, 
gas fraction in the disk).

We can accomodate both: we just define a classmethod 
```python
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
```

So that the class can be initialized both as 

```python
agn_env = AGNDiskEnv.from_smbh_parameters(
    alpha=0.1,
    m_smbh=1e7 * const.msun,
    r_from_smbh=0.1 * const.parsec,
    h=0.01,
    f_g=0.1,
)
```

or as 

```python
agn_env = AGNDiskEnv(alpha=0.1, cs=655897.7261203446, sigmag=129.06456913445584)
```

#### `numba` acceleration

This code is quite fast even though it's using a lot of `python` heavy machinery.
This is accomplished by decorating the longest-to-compute functions with `@njit`, 
so that they are compiled the first time they are executed.

#### `@property` interface

Some things are convenient to store one way and view another way. 
For example, the age `bbh.t` is stored in cgs units (seconds), but
it is more convenient to view it in megayears, since that's closer to the
typical scale of the temporal evolution of these binaries.

The `CompactBinary` `bbh` has an attribute `bbh.t`, and if we want to view it in megayears 
we can access `bbh.time_Myr`. These will always be in sync: how?
The property is defined as 

```python
    @property
    def time_Myr(self) -> float:
        return self.t / const.Myr
```

