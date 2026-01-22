
import torch
import torch.nn.functional as F

def multi_positive_loss(embeddings, group_size, temperature):
    z = F.normalize(embeddings, dim=1)

    sim_matrix = torch.matmul(z, z.T) / temperature     # [batch_size, batch_size]

    batch_size = embeddings.shape[0]
    mask = torch.zeros((batch_size, batch_size), device=embeddings.device)

    for i in range(0, batch_size, group_size):
        mask[i:i+group_size, i:i+group_size] = 1

    logits_mask = torch.scatter(
        torch.ones_like(mask, device=embeddings.device),                  # [batch_size, batch_size] meretu 1-esekbol allo mtx
        1,                                      # dim=1, soronkent 
        torch.arange(batch_size).view(-1, 1).to(embeddings.device),   # oszlopindex
        0                                       # nullat irunk a diagonalisba
    ).to(embeddings.device)                     # logits_mask: 1-esekbol allo matrix ahol a diagonalisok 0-ak
    mask = mask * logits_mask                   # mask: diagonalisban 10x10-es blokkok 1-esekkel feltoltve, de a diagonalisok 0-ak

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()       # numerikus stabilitas miatt

    exp_logits = torch.exp(logits) * logits_mask    # szimilaritasok exp-re emelve, a diagonalisok kinullazva
    log_sum_exp_denominator = torch.log(exp_logits.sum(1, keepdim=True))    # a fenti sorosszege egy vektorban
    log_prob = logits - log_sum_exp_denominator     # logaritmus argumentumaban osztas kivonaskent

    # log_prob: (i,j) eleme azt a valsz-et adja, hogy amennyiben i az anchor, j pozitiv

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)     # diagonalisakat ujbol kinullazzuk, sorosszeg, normalva a pozitivak szamaval

    loss = -mean_log_prob_pos.mean()
    return loss