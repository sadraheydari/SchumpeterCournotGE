# SchumpeterCournot

Dynamic Schumpeterian economy with Cournot competition and strategic R&D
investment.

This package implements a dynamic oligopoly model in which firms compete
in quantities (Cournot) and invest in R&D to improve productivity. The
model is solved using Policy Function Iteration (PFI).

------------------------------------------------------------------------

## Model Description

The model features:

-   Schumpeterian innovation
-   Cournot quantity competition
-   Strategic R&D investment
-   Endogenous productivity dynamics
-   No bond market (households consume current income)

Firms choose R&D investment to maximize expected discounted profits.
Innovation outcomes shift relative productivity and affect future
competition.

The state space tracks firm productivity levels and innovation outcomes
over time.

------------------------------------------------------------------------

## Installation

Clone the repository:

``` bash
git clone https://github.com/YOUR_USERNAME/SchumpeterCournot.git
cd SchumpeterCournot
```

Activate the project environment:

``` bash
julia --project=.
```

Then inside Julia:

``` julia
using Pkg
Pkg.instantiate()
```

------------------------------------------------------------------------

## Basic Usage

``` julia
using SchumpeterCournot

model = ModelSettings(
    τ_max = 10,
    max_iter_PF_solver = 50
)

solve_PFI!(model)
```

After solving:

-   `model.V_grid` contains the value function
-   `model.policy_grid` contains optimal R&D investment

------------------------------------------------------------------------

## Project Structure

    SchumpeterCournot/
    │
    ├── src/
    │   ├── SchumpeterCournot.jl
    │   ├── Types.jl
    │   ├── Bellman.jl
    │   ├── Solver.jl
    │   ├── ModelParams.jl
    │   ├── EconFuncs.jl
    │   ├── StateIndex.jl
    │   └── ProgressTrack.jl
    │
    ├── test/
    │   └── runtests.jl
    │
    ├── experiments/
    │   └── *.ipynb
    │
    ├── Project.toml
    └── Manifest.toml (optional)

-   `src/` contains the model implementation
-   `test/` contains structural and numerical tests
-   `experiments/` contains notebooks for calibration and simulation

------------------------------------------------------------------------

## Running Tests

From the project root:

``` bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

------------------------------------------------------------------------

## Running with Threads

To enable multi-threading:

``` bash
julia --project=. -t auto
```

Check thread count:

``` julia
Threads.nthreads()
```

------------------------------------------------------------------------

## Reproducibility

The `Project.toml` file specifies all dependencies.

To reproduce the environment:

``` julia
using Pkg
Pkg.instantiate()
```

If `Manifest.toml` is included, exact versions are restored.

------------------------------------------------------------------------

## Versioning

This project follows Semantic Versioning:

    MAJOR.MINOR.PATCH

Git tags correspond to versions in `Project.toml`.

------------------------------------------------------------------------

## Author

Mohammad Sadra Heydari\
University of Glasgow

------------------------------------------------------------------------

## License

Specify license here (e.g., MIT License).
