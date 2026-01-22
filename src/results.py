# %%
# VISUALIZATION
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from torch.utils.data import DataLoader
from data_loader import make_dloaders
from model import ChessEncoder
from consts import *
from train import ChessTrainingModule
#%%
FILE_PATH = "../data/lc0-hidden/lichess_elite_2025-11.h5"
CHECKPOINT_PATH = '../lightning_logs/version_19/checkpoints/epoch=2-step=51428.ckpt' 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE_VIZ = 6 # Hány játékot vizualizáljunk összesen? (1 anchor + 5 negatív)

print(f"Vizualizáció előkészítése {DEVICE} eszközön...")

_, _, loader_viz = make_dloaders(FILE_PATH)
viz_states_batch = next(iter(loader_viz))[:BATCH_SIZE_VIZ]
# Kilapítjuk: [6, 10, 64, 8, 8] -> [60, 64, 8, 8]
viz_states_flat = viz_states_batch.view(-1, *INPUT_SIZE).to(DEVICE)

# Az első 10 pont a "Game 0" (Anchor), a következő 10 a "Game 1", stb.
labels = np.repeat(np.arange(BATCH_SIZE_VIZ), 10)
colors = ['red' if l == 0 else 'blue' if l == 1 else 'gray' for l in labels]
sizes = [100 if l == 0 else 50 if l == 1 else 20 for l in labels]
alphas = [1.0 if l == 0 else 0.7 if l == 1 else 0.3 for l in labels]

# A) Tanítatlan modell (véletlen súlyokkal)
model_untrained = ChessEncoder(embedding_dim=ENCODER_EMBEDDING_DIM).to(DEVICE)
model_untrained.eval()

# B) Betanított modell (betöltjük a súlyokat)
try:
    model_trained = ChessTrainingModule.load_from_checkpoint(CHECKPOINT_PATH).to(DEVICE)
    print("Betanított modell súlyai sikeresen betöltve.")
except FileNotFoundError:
    print(f"HIBA: Nem találom a {CHECKPOINT_PATH} fájlt! Fuss le a tanítást előbb.")
model_trained.eval()
#%%

# GENERATING EMBEDDINGS & T-SNE
def get_embeddings_and_tsne(model, inputs):
    with torch.no_grad():
        embeddings = model(inputs).cpu().numpy()
    
    # perplexity: a szomszédság mérete, kis adathoz kisebb érték kell
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
        Line2D([0], [0], marker='o', color='w', label='Anchor Játék (Pozitívok)', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Negatív Játék 1', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Többi Negatív', markerfacecolor='gray', markersize=6),
    ]
    ax.legend(handles=legend_elements, loc='best')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

plot_embeddings(emb_untrained_2d, "TANÍTÁS ELŐTT (Véletlen reprezentáció)", ax1)
plot_embeddings(emb_trained_2d, "TANÍTÁS UTÁN (Kontrasztív reprezentáció)", ax2)

plt.suptitle(f"A sakktábla-állások reprezentációjának fejlődése - TESZTADATON\n(Loss: ~4.2?)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()