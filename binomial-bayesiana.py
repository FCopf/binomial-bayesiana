from pathlib import Path
import shinyswatch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, beta
from scipy.integrate import simpson
from shiny import App, ui, render, reactive

# Define o diretório do aplicativo (onde o arquivo do app está localizado)
app_dir = Path(__file__).parent

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="styles.css")
    ),
    ui.panel_title("Inferência Bayesiana Binomial"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.card(
                ui.card_header("Dados do experimento", class_="card-header-dados"),
                ui.input_numeric("N_value", "Número de observações (N)", value=10, min=1),
                ui.output_ui("k_slider"),
            ),
            ui.card(
                ui.card_header("Priori", class_="card-header-priori"),
                ui.input_slider("alpha_prior", "Parâmetro α", min=1, max=20, value=1, step=0.1),
                ui.input_slider("beta_prior", "Parâmetro β", min=1, max=20, value=1, step=0.1),
            ),
            
            ui.card(
                ui.card_header("Inferência", class_="card-header-inferencia"),
                ui.input_slider("x1_value", "Limite inferior (x1)", min=0.0, max=1.0, value=0.2, step=0.01),
                ui.input_slider("x2_value", "Limite superior (x2)", min=0.0, max=1.0, value=0.8, step=0.01),
                
            ),
        ),
        ui.output_plot("plot_posterior", height="800px")
    ),
    theme=shinyswatch.theme.darkly,
)

# Server
def server(input, output, session):
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
        x1 = input.x1_value()
        x2 = input.x2_value()
        alpha_param = input.alpha_prior()
        beta_param = input.beta_prior()
        
        p_grid = np.linspace(0, 1, 1000)
        prior = beta.pdf(p_grid, a=alpha_param, b=beta_param)
        likelihood = binom.pmf(k, N, p_grid)
        posterior_unnormalized = likelihood * prior
        posterior = posterior_unnormalized / simpson(y=posterior_unnormalized, x=p_grid)
        prior_prob = beta.cdf(x2, a=alpha_param, b=beta_param) - beta.cdf(x1, a=alpha_param, b=beta_param)
        post_prob = beta.cdf(x2, a=alpha_param + k, b=beta_param + N - k) - beta.cdf(x1, a=alpha_param + k, b=beta_param + N - k)


        # Figura com 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(8, 9))

        # Prior
        axs[0].plot(p_grid, prior, color='red')
        axs[0].fill_between(p_grid, prior, where = ((p_grid >= x1) & (p_grid <= x2)), color='red', alpha = 0.2)
        axs[0].set_title(f"Priori Beta - P({x1:.2f} ≤ p ≤ {x2:.2f}) = {prior_prob:.2f}")
        axs[0].set_ylabel("Densidade")
        axs[0].grid(True)
        
        # Verossimilhança
        axs[1].plot(p_grid, likelihood, color='green')
        axs[1].set_title(f"Verossimilhança (Binomial) para k={k}, N={N}")
        axs[1].set_ylabel("Densidade")
        axs[1].grid(True)

        # Posterior
        axs[2].plot(p_grid, posterior, color='blue')
        axs[2].fill_between(p_grid, posterior, where = ((p_grid >= x1) & (p_grid <= x2)), color='blue', alpha = 0.2)
        axs[2].set_title(f"Posteriori (Priori × Verossimilhança) - P({x1:.2f} ≤ p ≤ {x2:.2f}) = {post_prob:.2f}")
        axs[2].set_xlabel("p (taxa de sucesso)")
        axs[2].set_ylabel("Densidade")
        axs[2].grid(True)

        plt.tight_layout()
        return fig

# Cria o aplicativo Shiny e serve arquivos estáticos a partir da pasta www
app = App(app_ui, server, static_assets=app_dir / "www")
