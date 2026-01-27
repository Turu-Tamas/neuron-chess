# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from torch.utils.data import DataLoader
from data_loader import make_dloaders
from model import ChessEncoder
from consts import *

#%%
FILE_PATH = "../data/lc0-hidden/lichess_elite_2025-11.h5"
CHECKPOINT_PATH = '../lightning_logs/version_19/checkpoints/epoch=2-step=51428.ckpt' 
MODEL_PATH = '../weights/v13_temp0.07_emb32.pt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VISUALIZATION_GAME_COUNT = 6 

_, _, loader_viz = make_dloaders(FILE_PATH)
viz_states_batch = next(iter(loader_viz))[:VISUALIZATION_GAME_COUNT]
# [6, 10, 64, 8, 8] -> [60, 64, 8, 8]
viz_states_flat = viz_states_batch.view(-1, *INPUT_SIZE).to(DEVICE)

# az elso 10 pont a 0-as label, a kovetkezo 10 az 1-es label...
labels = np.repeat(np.arange(VISUALIZATION_GAME_COUNT), 10)
colors = ['red' if l == 0 else 'blue' if l == 1 else 'gray' for l in labels]
sizes = [100 if l == 0 else 50 if l == 1 else 20 for l in labels]
alphas = [1.0 if l == 0 else 0.7 if l == 1 else 0.3 for l in labels]

# tanitatlan modell veletlen sulyokkal
model_untrained = ChessEncoder(embedding_dim=ENCODER_EMBEDDING_DIM).to(DEVICE)
model_untrained.eval()

# betanitott modell
model_trained = ChessEncoder(embedding_dim=128).to(DEVICE)
try:
    model_trained.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Betanított modell súlyai sikeresen betöltve.")
except FileNotFoundError:
    print(f"HIBA: Nem találom a {MODEL_PATH} fájlt! Fuss le a tanítást előbb.")
model_trained.eval()
#%%

# GENERATING EMBEDDINGS & T-SNE
def get_embeddings_and_tsne(model, inputs):
    with torch.no_grad():
        embeddings = model(inputs).cpu().numpy()
    
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=POSITIONS_PER_GAME, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d

print("t-SNE futtatása a tanítatlan modellen...")
emb_untrained_2d = get_embeddings_and_tsne(model_untrained, viz_states_flat)

print("t-SNE futtatása a betanított modellen...")
emb_trained_2d = get_embeddings_and_tsne(model_trained, viz_states_flat)


# VISUALIZATION
def plot_embeddings(embeddings_2d, title, ax):
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=sizes, alpha=alphas)

    anchor_points = embeddings_2d[0:POSITIONS_PER_GAME]
    centroid = np.mean(anchor_points, axis=0)
    for point in anchor_points:
        ax.plot([centroid[0], point[0]], [centroid[1], point[1]], 'r-', alpha=0.3)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Anchor játék (Pozitívak)', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Negatív játék 1', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Többi negatív', markerfacecolor='gray', markersize=6),
    ]
    ax.legend(handles=legend_elements, loc='best')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

plot_embeddings(emb_untrained_2d, "TANÍTÁS ELŐTT (Véletlen reprezentáció)", ax1)
plot_embeddings(emb_trained_2d, "TANÍTÁS UTÁN (Kontrasztív reprezentáció)", ax2)

plt.suptitle(f"A sakktábla-állások reprezentációjának fejlődése - TESZTADATON\n(Loss: ~4.2?)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
