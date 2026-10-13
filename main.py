import torch

from src.dataset import Scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scheduler = Scheduler(
    generator=torch.Generator(device=device).manual_seed(42),
    training_steps=1000,
    beta_start=1e-4,
    beta_end=0.02,
)

# Simula un'immagine normalizzata nell'intervallo [-1, 1].
x0 = torch.rand(
    (1, 3, 32, 32),
    device=device,
) * 2 - 1

# Scegli un timestep elevato.
t = 980
timesteps = torch.tensor([t], device=device)

# Genera il campione rumoroso e conserva il rumore esatto.
xt, noise = scheduler.add_noise(x0, timesteps)

# Ricostruisci x0 usando il rumore esatto.
alpha_bar = scheduler.alphas_cumprod[t].to(device)

x0_reconstructed = (
    xt - torch.sqrt(1 - alpha_bar) * noise
) / torch.sqrt(alpha_bar)

print("Original min/max:", x0.min().item(), x0.max().item())
print(
    "Reconstructed min/max:",
    x0_reconstructed.min().item(),
    x0_reconstructed.max().item(),
)
print(
    "Max reconstruction error:",
    (x0_reconstructed - x0).abs().max().item(),
)