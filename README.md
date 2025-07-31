# Astrocyte-neuron_BRIAN2-network

This repository contains the code used for the simulations presented in *"A biologically plausible model of astrocyte-neuron networks in random and hub-driven connectivity"*  
by **Salzano G., Paradisi P., and Cataldo E.**

Simulations were conducted using the [Brian2 simulator](https://github.com/brian-team/brian2). Our computational model can be considered a refinement of Chapter 18 in the book *"Computational Glioscience"* edited by M. De Pittà and H. Berry (Springer, 2019) — [GitHub repository of the book](https://github.com/mdepitta/comp-glia-book?tab=readme-ov-file#ch18stimberg).

## Repository contents

- A `.py` file containing the complete computational model and all parameters used in the final simulations described in the paper.
- A `.tar.xz` archive with the compressed adjacency matrix used to implement the hub-driven neuronal connectivity.  
  **To run the simulations, please extract and use the `adjacency_matrix.txt` file.**

## Data visualization and analysis

For data visualization and further analysis, we recommend referring to:

- [Brian2 team GitHub](https://github.com/brian-team)
- [Marcel Stimberg's GitHub](https://github.com/mstimberg)
- [Maurizio De Pittà's comp-glia-book repository](https://github.com/mdepitta/comp-glia-book)
