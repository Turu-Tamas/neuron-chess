# Chess Encoder

In this project, we created an encoder mapping chess boards to a learned representation.
We used Leela chess zero (a neural network chess engine) as a feature extractor and
trained a small network on top of that with contrastive learning to map boards from
within a game to similar embeddings and boards from different games to different representations.

These are t-SNE projections of the embeddings of 10 boards sampled from 10 different games.
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/8272776d-0fd8-4823-9586-0b9cc2d7476f" />
