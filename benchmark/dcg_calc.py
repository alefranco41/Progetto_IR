import math

def calculate_dcg(scores):
    dcg = scores[0]  # Il primo elemento non viene scontato
    for i in range(1, len(scores)):
        dcg += scores[i] / math.log2(i + 1)  # Applica uno sconto di log2(i+1)
    return dcg

def calculate_idcg(scores):
    scores_sorted = sorted(scores, reverse=True)
    return calculate_dcg(scores_sorted)

def calculate_ndcg(scores):
    dcg = calculate_dcg(scores)
    idcg = calculate_idcg(scores)
    if idcg == 0:
        return 0  # Per evitare divisione per zero
    return dcg / idcg

# Punteggi delle recensioni (in base a Review Rating)
scores = [3,3,1,0,3,3,3,0,3,2]

# Calcola il DCG
dcg_score = calculate_dcg(scores)
print("DCG Score:", dcg_score)

# Calcola l'IDCG
idcg_score = calculate_idcg(scores)
print("IDCG Score:", idcg_score)

# Calcola l'nDCG
ndcg_score = calculate_ndcg(scores)
print("nDCG Score:", ndcg_score)
