"""
Particle Swarm Optimization (PSO)

Copyright (c) 2020 Gabriele Gilardi

References
----------
- Maurice Clerc, 2012, "Standard Particle Swarm Optimisation", hal-00764996,
  at https://hal.archives-ouvertes.fr/hal-00764996/document.
- Maurice Clerc, 2006, "Particle Swarm Optimisation", ISTE Ltd, at
  https://onlinelibrary.wiley.com/doi/book/10.1002/9780470612163.
- Martin Roberts, 2020, "How to generate uniformly random points on n-spheres
  and in n-balls", Extreme Learning, at http://extremelearning.com.au

Characteristics
---------------
- The code has been written and tested in Python 3.7.7.
- Particle Swarm Optimization (PSO) implementation for minimization problems.
- Variables can be real, integer, or mixed real/integer.
- Confidence coefficients depend on one single parameter.
- Search space can be normalized to improve convergency.
- An adaptive random topology is used to define each agent's neighbourhood
  (with an option to use the full swarm as neighbourhood).
- Unbiased velocity equation using hyperspherical uniform distribution
  (including the corner case where an agent is the neighbourhood best).
- Three velocity confinement methods (hyperbolic, random-back, and mixed
  hyperbolic/random-back).
- Possibility to specify the velocity limits.
- To improve the execution speed the algorithm has been designed without any
  loop on the agents.
- An arbitrary number of parameters can be passed (in a tuple) to the function
  to minimize.
- Option to run sequential tests with constant or random (uniformly distributed)
  number of agents.
- Four examples are included: Parabola, Alpine, Tripod, and Ackley.
- Usage: python test.py.

Parameters
----------
func
    Function to minimize. The position of all agents is passed to the function
    at the same time.
LB, UB
    Lower and upper boundaries of the search space.
nPop >=1, epochs >= 1
    Number of agents (population) and number of iterations.
K >= 0
    Average size of each agent's group of informants. If K=0 the entire swarm
    is used as agent's group of informants.
phi >= 2
    Coefficient to calculate the self-confidence and the confidence-in-others
    coefficients.
vel_fact > 0
    Velocity factor to calculate the max. and min. allowed velocities.
conf_type = HY, RB, MX
    Confinement on the velocities: hyperbolic, random-back, mixed.
IntVar
    List of indexes specifying which variable should be treated as integer.
    If all variables are real set IntVar=None, if all variables are integer
    set IntVar='all'. Indexes are in the range (1,nVar).
normalize = True, False
    Specifies if the search space should be normalized.
args
    Tuple containing any parameter that needs to be passed to the function. If
    no parameters are passed set args=None.
nVar
    Number of variables.
nRun
    Number of runs.
random_pop = True, False
    Specifies if the number of agents (for each run) is kept constant at nPop
    or randomly choosen from the uniform distribution nVar/2 <= nPop <= nVar/2.
Xsol
    Solution to the minimization point. Set Xsol=None if not known.

Examples
--------
There are four examples: Parabola, Alpine, Tripod, and Ackley.

- Parabola, Alpine, and Ackley can have an arbitrary number of dimensions,
  while Tripod has only two dimensions.

- Parabola, Tripod, and Ackley are examples where parameters (respectively,
  array X0, scalars kx and ky, and array X0) are passed using args.

- The global minimum for Parabola and Ackley is at X0; the global minimum for
  Alpine is at zero; the global minimum for Tripod is at [0,-ky] with local
  minimum at [-kx,+ky] and [+kx,+ky].
"""

import sys
import time
import numpy as np
from pso import PSO

# ======= Examples ======= #

# Un-comment the desired example
example = 'Parabola'
# example = 'Alpine'
# example = 'Tripod'
# example = 'Ackley'

# Parabola: F(X) = sum((X - X0)^2), Xmin = X0
if (example == 'Parabola'):

    def Parabola(X, args):
        X0 = args
        F = ((X - X0) ** 2).sum(axis=1)
        return F

    # Problem parameters
    nVar = 30
    nRun = 15
    random_pop = False
    X0 = np.ones(nVar) * 1.2345        # Args

    # PSO parameters
    func = Parabola
    UB = np.ones(nVar) * 20.0
    LB = - UB
    nPop = 40
    epochs = 500
    K = 3
    phi = 2.05
    vel_fact = 0.5
    conf_type = 'RB'
    IntVar = None
    normalize = False
    args = (X0)

    # Solution
    Xsol = X0

# Alpine: F(X) = sum(abs(X*sin(X) + 0.1*X)), Xmin = 0
elif (example == 'Alpine'):

    def Alpine(X, args):
        F = np.abs(X * np.sin(X) + 0.1 * X).sum(axis=1)
        return F

    # Function parameters
    nVar = 10
    nRun = 15
    random_pop = False

    # PSO parameters
    func = Alpine
    UB = np.ones(nVar) * 10.0
    LB = - UB
    nPop = 100
    epochs = 500
    K = 3
    phi = 2.05
    vel_fact = 0.5
    conf_type = 'RB'
    IntVar = None
    normalize = False
    args = None

    # Solution
    Xsol = np.zeros(nVar)

# Tripod:
# F(x,y)= p(y)*(1 + p(x)) + abs(x + kx*p(y)*(1 - 2*p(x)))
#         + abs(y + ky*(1 - 2*p(y)))
# p(x) = 1 if x >= 0, p(x) = 0 if x < 0; p(y) = 1 if y >= 0, p(y) = 0 if y < 0
# Global minimum at [0,-ky], local minimum at [-kx,ky] and [kx,ky]; kx, ky > 0
elif (example == 'Tripod'):

    def Tripod(X, args):
        x = X[:, 0]
        y = X[:, 1]
        kx = args[0]
        ky = args[1]
        px = (x >= 0.0)
        py = (y >= 0.0)
        F = py * (1.0 + px) + np.abs(x + kx * py * (1.0 - 2.0 * px)) \
            + np.abs(y + ky * (1.0 - 2.0 * py))
        return F

    # Function parameters
    nVar = 2                # The equation works only with two dimensions
    nRun = 15
    random_pop = False
    kx = 20.0               # Args
    ky = 40.0

    # PSO parameters
    func = Tripod
    UB = np.ones(nVar) * 100.0
    LB = - UB
    nPop = 40
    epochs = 500
    K = 3
    phi = 2.05
    vel_fact = 0.5
    conf_type = 'RB'
    IntVar = None
    normalize = False
    args = (kx, ky)

    # Solution
    Xsol = np.array([0.0, -ky])

# Ackley: F(X)= 20 + exp(1) - exp(sum(cos(2*pi*X))/n)
#               - 20*exp(-0.2*sqrt(sum(X^2)/n))
# Xmin = X0
elif (example == 'Ackley'):

    def Ackley(X, args):
        X0 = args
        n = float(X.shape[1])
        F = 20.0 + np.exp(1.0) \
            - np.exp((np.cos(2.0 * np.pi * (X - X0))).sum(axis=1) / n) \
            - 20.0 * np.exp(-0.2 * np.sqrt(((X - X0) ** 2).sum(axis=1) / n))
        return F

    # Function parameters
    nVar = 30
    nRun = 15
    random_pop = False
    X0 = np.ones(nVar) * 1.6            # args

    # PSO parameters
    func = Ackley
    UB = np.ones(nVar) * 30.0
    LB = - UB
    nPop = 40
    epochs = 500
    K = 3
    phi = 2.05
    vel_fact = 0.5
    conf_type = 'RB'
    IntVar = None
    normalize = True
    args = (X0)

    # Solution
    Xsol = X0

else:
    print("Function not found")
    sys.exit(1)

# ======= Main Code ======= #

np.random.seed(1294404794)          # Seed random generator

# Define number of agents for each run
if (random_pop):
    delta = nVar // 2
    agents = np.random.randint(nPop - delta, nPop + delta, size=nRun)
else:
    agents = np.ones(nRun, dtype=int) * nPop

print()
print("Function: ", example)

best_pos = np.zeros((nRun, nVar))
best_cost = np.zeros(nRun)
t_start = time.perf_counter()

# Run cases
for run in range(nRun):

    # Optimize this run
    nPop = agents[run]
    X, F = PSO(func, LB, UB, nPop=nPop, epochs=epochs, K=K, phi=phi,
               vel_fact=vel_fact, conf_type=conf_type, IntVar=IntVar,
               normalize=normalize, args=args)

    # Save best position/cost for each run
    best_pos[run, :] = X
    best_cost[run] = F

    # Print run number, agents number, and the error
    if (Xsol is not None):
        print("run = {0:<3d}     nPop = {1:<3d}     Error: {2:e}"
              .format(run+1, nPop, np.linalg.norm(Xsol - X)))
    else:
        print("run = {0:<3d}     nPop = {1:<3d}".format(run+1, nPop))

t_end = time.perf_counter()

# Results
best_pos_mean = best_pos.mean(axis=0)
best_pos_std = best_pos.std(axis=0)
print()
print("Best position:")
for var in range(nVar):
    print("var = {0:<3d}     mean = {1: e}     std = {2:e}"
          .format(var+1, best_pos_mean[var], best_pos_std[var]))
print()
print("Best cost:    mean = {0: e}     std = {1:e}"
      .format(best_cost.mean(), best_cost.std()))
print()
if (Xsol is not None):
    error = np.linalg.norm(Xsol - best_pos_mean)
    print("Error: {0:e}".format(error))
print()
print("Total run time = {0:<7.2f}".format(t_end - t_start))
