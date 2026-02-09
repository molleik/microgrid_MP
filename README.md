# Cleaner energy microgrids under market power and limited regulation in developing countries

## Project description
The model proposed in this repository, and its corresponding paper, are part of a project titled "From Off-grid Solar PV Systems to Microgrids in Lebanon". 

### Context
Under the instability and unavailability of their national grid, consumers in developing countries have invested in their own generation solutions. These commonly include a neighborhood diesel generator serving a grid parallel to the national one (microgrid), and household rooftop solar photovoltaic (PV) systems. Such setups have their shortcomings. The owner of the neighborhood diesel generator, referred to as the diesel generator company (DGC), has uncontested market power over the microgrid: it decides on the access, grid availability, investments and tariffs within the microgrid under limited regulatory oversight. Moreover, household PV-owners suffer from the intermittent nature of their renewable systems, wasting excess generation potential during hours of high solar capacity factors.

### Purpose and research questions
This project investigates the feasibility and impact of introducing renewable energy to the microgrid through a DGC-owned PV and battery system, as well as feed-in from household PV owners' (prosumers) electric generation potential. Specifically, the research question to be answered is: _How can a regulator, with limited regulatory oversight and who cannot challenge the profitability of the entrenched DGC, design a policy that increases the utilization of the existing household PV assets, reduces reliance on diesel-based generation, limits unmet demand and improves the affordability of electricity in a DGC-controlled microgrid?_
This work also quantifies the effect of limited regulation by proposing a theoretical regulator with extended control over the microgrid.

### Data and Case Study
This model is applied to the case of a microgrid in Deir Qanoun Ennaher, South Lebanon. High-resolution data has been logged in the existing microgrid for this study.

## Repository description
This repository presents the code for a bi-level game theoretic model, with a weak regulator maximizing household economic surplus (HES) at the first level, and the profit-maximizing DGC at the second. 
The `main` branch of the repository is structured as follows:
<pre>
Model-1/
├── Inputs/                                      # Input files
│ ├── inputs.xlsx/                               # Inputs for the microgrid at the status quo
│ └── inputs_RE.xlsx/                            # Inputs for the microgrid after RE penetration
├── Outputs/                                     # Output files
│ ├── 0. Current Case/                           # Output file at the status quo
│ ├── 1. Budget/                                 # Output under different budget constraints
| ├── 2. RE target/                              # Output under different RE targets
│ ├── 3. Prosumer percentage/                    # Output for different microgrids with different prosumer percentages
| ├── 4. Prosumer percentage constrained/        # Output under different budget constraints and extended regulation
│ └── 5. Budget constrained/                     # Outputs for different microgrids with different prosumer percentages and extended regulation
├── `functions.py`                               # Output generating functions
├── `main.py`                                    # Run
├── `model_1.py`                                 # Microgrid class
├── `plot_functions.py`                          # Output visualization functions
├── README.md                                    # Project documentation
└── LICENSE                                      # License information
</pre>
