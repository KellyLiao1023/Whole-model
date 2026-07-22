import pickle
import numpy as np
import pandas as pd

def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def Params2PSAM(Params, mode='array'):
    
    SeqLen = int(len(Params)/3)
    Params = Params.reshape(SeqLen, 3)
    Params = np.concatenate((Params, -Params.sum(axis=1).reshape(SeqLen, 1)), axis=1)
    
    if mode == 'array': return Params
    elif mode == 'dataframe': return pd.DataFrame(Params, columns=['A', 'C', 'G', 'T'])
    else: return None

def GBI_numba(arr_seq, arr_zeros, positions, SeqLen):
    L = len(positions)
    for i, p in enumerate(positions):
        arr_zeros += arr_seq//4**(SeqLen-1-p)%4*4**(L-1-i)
    return arr_zeros

def groupby_index(positions, SeqLen=10):
    arr_seq = np.arange(4**SeqLen)
    arr_zeros = np.zeros(4**SeqLen)
    return GBI_numba(arr_seq, arr_zeros, np.array(positions), SeqLen).astype(int)

def Motif2Seqs(motif):
    # transfer a motif to all possible sequences
    # ex: 'ANCG' -> ['AACG', 'ACCG', 'AGCG', 'AUCG']
    pools = [tuple(pool) for pool in [IUPAC_nucleotide(m) for m in motif]]
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield ''.join(prod)

def IUPAC_nucleotide(var, mode='DNA'):
    if mode == 'RNA':
        return {'Z': '', 
                'A': 'A', 
                'C': 'C', 
                'G': 'G', 
                'T': 'U', 
                'U': 'U', 
                'W': 'AU', 
                'S': 'CG', 
                'M': 'AC', 
                'K': 'GU', 
                'R': 'AG', 
                'Y': 'CU', 
                'B': 'CGU', 
                'D': 'AGU', 
                'H': 'ACU', 
                'V': 'ACG', 
                'N': 'ACGU', 
                'n': 'ACGU', 
                '_': '_', 
                '.': '.'}.get(var, None)
    elif mode == 'DNA':
        return {'Z': '', 
                'A': 'A', 
                'C': 'C', 
                'G': 'G', 
                'T': 'T', 
                'U': 'T', 
                'W': 'AT', 
                'S': 'CG', 
                'M': 'AC', 
                'K': 'GT', 
                'R': 'AG', 
                'Y': 'CT', 
                'B': 'CGT', 
                'D': 'AGT', 
                'H': 'ACT', 
                'V': 'ACG', 
                'N': 'ACGT', 
                'n': 'ACGT', 
                '_': '_', 
                '.': '.'}.get(var, None)

def Matrix2Score(matrix):
    scores = np.array([matrix.values[i][groupby_index([i], 6)] for i in range(6)]).sum(axis=0)
    return pd.DataFrame(scores, index=list(Motif2Seqs('N'*6))).to_dict()[0]

def getPFM(sequences, ratios=False, pseudocount=0, counts=None):
    """
    Calculates the position frequency matrix (PFM) or matrix of ratios for a set of DNA sequences.

    Args:
    sequences (List[str]): A list of DNA sequences.
    counts (List[int], optional): A list of counts representing the number of times each sequence occurs.
        Defaults to None, in which case all counts are set to 1.
    ratios (bool, optional): If True, returns the matrix of ratios instead of the PFM.
        Defaults to False.
    pseudocount (int, optional): The value of the pseudocount to be added to each element in the PFM.
        Defaults to 1.

    Returns:
    pd.DataFrame: A DataFrame representing the PFM or matrix of ratios.
    """
    sequences = [x for x in sequences if not "N" in x]
    
    if counts is None:
        counts = [1] * len(sequences)

    n = len(sequences[0])  # length of sequences
    pfm = np.zeros((n, 4))  # initialize PFM with zeros

    nucleotide_index = {"A": 0, "C": 1, "G": 2, "T": 3}

    # loop over each sequence and update the PFM
    for seq, count in zip(sequences, counts):
        seq_indices = np.array([nucleotide_index[n] for n in seq])
        pfm[np.arange(n), seq_indices] += count

    # Add pseudocounts to the PFM
    pfm += pseudocount

    # convert PFM or matrix of ratios to DataFrame
    pfm_df = pd.DataFrame(pfm, columns=["A", "C", "G", "T"])

    # calculate the matrix of ratios if requested
    if ratios:
        return pfm_df / (np.sum(counts) + pseudocount * 4)
    else:
        return pfm_df