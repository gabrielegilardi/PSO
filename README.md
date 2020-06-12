# Metaheuristic Minimization Using Particle Swarm Optimization

## Reference

- Maurice Clerc, 2012, "[Standard Particle Swarm Optimisation](https://hal.archives-ouvertes.fr/hal-00764996/document)", hal-00764996.
- Maurice Clerc, 2006, "[Particle Swarm Optimisation](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470612163)", ISTE Ltd.
- Martin Roberts, 2020, ["How to generate uniformly random points on n-spheres and in n-balls"](http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/), Extreme Learning.

## Characteristics

- The code has been written and tested in Python 3.7.7.
- Particle Swarm Optimization (PSO) implementation for metaheuristic minimization.
- Variables can be real, integer, or mixed real/integer.
- Variables can be constrained to a specific interval or value setting the lower and the upper boundaries.  
- Confidence coefficients depend on one single parameter.
- Search space can be normalized to improve convergency.
- An adaptive random topology is used to define each agent's neighbourhood (with an option to use the full swarm as neighbourhood).
- Unbiased velocity equation using hyperspherical uniform distribution (including the corner case where an agent is the neighbourhood best).
- Three velocity confinement methods (hyperbolic, random-back, and mixed hyperbolic/random-back).
- Possibility to specify the velocity limits.
- To improve the execution speed the algorithm has been designed without any loop on the agents.
- An arbitrary number of parameters can be passed (in a tuple) to the function to minimize.
- Option to run sequential tests with constant or random (uniformly distributed) number of agents.
- Usage: *python test.py example*.

## Parameters

`example` Name of the example to run (Parabola, Alpine, Tripod, and Ackley.)

`func` Function to minimize. The position of all agents is passed to the function at the same time.

`LB`, `UB` Lower and upper boundaries of the search space.

`nPop`, `epochs` Number of agents (population) and number of iterations.

`K` Average size of each agent's group of informants. If `K=0` the entire swarm is used as agent's group of informants.

`phi` Coefficient to calculate the self-confidence coefficient and the confidence-in-others coefficient.

`vel_fact` Velocity factor to calculate the maximum and the minimum allowed velocities.

`conf_type` Confinement type (on the velocities): `HY=` hyperbolic, `RB=` random-back, `MX=` mixed hyeperbolic/random-back.

`IntVar` List of indexes specifying which variable should be treated as integer. If all variables are real set `IntVar=None`, if all variables are integer set `IntVar=all`. Indexes are in the range `(1,nVar)`.

`normalize` Specifies if the search space should be normalized (to improve convergency).

`rad` Normalized radius of the hypersphere centered on the best particle. The higher the number of other particles inside and the better is the solution.

`args` Tuple containing any parameter that needs to be passed to the function to minimize. If no parameters are passed set `args=None`.

`nVar` Number of variables (dimensions of the search space).

`nRun` Number of runs for a specific case.

`random_pop` Specifies if the number of agents (for each run) is kept constant at `nPop` or randomly choosen from the uniform distribution `nVar/2 <= nPop <= nVar/2`.

`Xsol` Solution to the minimization point. Set to `None` if not known.

## Examples

There are four examples: Parabola, Alpine, Tripod, and Ackley (see *test.py* for the specific equations and parameters). As illustration, a 3D plot of these function is shown [here](examples.bmp).

- **Parabola**, **Alpine**, and **Ackley** can have an arbitrary number of dimensions, while **Tripod** has only two dimensions.

- **Parabola**, **Tripod**, and **Ackley** are examples where parameters (respectively, array `X0`, scalars `kx` and `ky`, and array `X0`) are passed using `args`.

- The global minimum for **Parabola** and **Ackley** is at `X0`; the global minimum for **Alpine** is at zero; the global minimum for **Tripod** is at `[0,-ky]` with local minimum at `[-kx,+ky]` and `[+kx,+ky]`.
