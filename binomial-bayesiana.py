import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, beta
from scipy.integrate import simpson
from shiny import App, ui, render, reactive

# Interface do usuário
app_ui = ui.page_fluid(
    ui.panel_title("Inferência Bayesiana: priori, verossimilhança e posteriori"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Dados do experimento"),
            ui.input_numeric("N_value", "Número de observações (N)", value=10, min=1),
            ui.output_ui("k_slider"),
            ui.h4("Priori"),
            ui.input_slider("alpha_prior", "Parâmetro α", min=1, max=20, value=1, step=0.1),
            ui.input_slider("beta_prior", "Parâmetro β", min=1, max=20, value=1, step=0.1)
        ),
        ui.output_plot("plot_posterior", height="800px")  # Increased height to fit three plots
    )
)

# Lógica do servidor
def server(input, output, session):
    # Dynamic UI for k slider based on N
    @output
    @render.ui
    def k_slider():
        N = input.N_value()
        return ui.input_slider("k_value", "Número de sucessos (k)", min=0, max=N, value=min(6, N))
    
    @output
    @render.plot
    def plot_posterior():
        N = input.N_value()
        k = input.k_value()
        alpha = input.alpha_prior()
        beta_param = input.beta_prior()
        
        p_grid = np.linspace(0, 1, 1000)
        
        prior = beta.pdf(p_grid, a=alpha, b=beta_param)
        likelihood = binom.pmf(k, N, p_grid)
        posterior_unnormalized = likelihood * prior
        posterior = posterior_unnormalized / simpson(posterior_unnormalized, p_grid)

        # Create a figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(8, 9))

        # Plot prior
        axs[0].plot(p_grid, prior, color='red')
        axs[0].set_title(f"Priori Beta({alpha:.2f}, {beta_param:.2f})")
        axs[0].set_ylabel("Densidade")
        axs[0].grid(True)

        # Plot likelihood
        axs[1].plot(p_grid, likelihood, color='green')
        axs[1].set_title(f"Verossimilhança (Binomial) para k={k}, N={N}")
        axs[1].set_ylabel("Densidade")
        axs[1].grid(True)

        # Plot posterior
        axs[2].plot(p_grid, posterior, color='blue')
        axs[2].set_title(f"Posteriori (Priori × Verossimilhança)")
        axs[2].set_xlabel("p (taxa de sucesso)")
        axs[2].set_ylabel("Densidade")
        axs[2].grid(True)

        plt.tight_layout()
        return fig

# Criação do app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()