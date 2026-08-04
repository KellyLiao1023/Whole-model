import pickle
import numpy as np
import pandas as pd


def save_object(obj, filename):
    # Open a file for writing
    with open(filename, 'wb') as f:
        # Use pickle.dump to save the object to the file
        pickle.dump(obj, f)

def load_object(filename):
    # Open the file for reading
    with open(filename, 'rb') as f:
        # Use pickle.load to load the object from the file
        obj = pickle.load(f)
    return obj

def write_to_file(file_path, content, mode='w'):
    with open(file_path, mode) as file:
        file.write(content)
    return None

# def load_obj(filename):
#     with open(filename, 'rb') as f:
#         return pickle.load(f)

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

import pickle
import numpy as np
import pandas as pd

def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_obj(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

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


def PWM2Score(matrix):
    '''
    Computes scores for all possible sequences generated from a position weight matrix

    Args:
        matrix (pandas.DataFrame): a position weight matrix

    Returns:
        dict: a dictionary of sequence to score pairs
    '''
    n = len(matrix)
    scores = np.array([matrix.values[i][groupby_index([i], n)] for i in range(n)]).sum(axis=0)
    return pd.DataFrame(scores, index=list(Motif2Seqs('N'*n))).to_dict()[0]


def Motif2Seqs(motif, mode='DNA'):
    '''
    transfer a motif to all possible sequences
    ex: 'ANCG' -> ['AACG', 'ACCG', 'AGCG', 'AUCG']
    '''
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
                    'n': 'ACGU'}.get(var, var)
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
                    'n': 'ACGT'}.get(var, var)
    
    pools = [tuple(pool) for pool in [IUPAC_nucleotide(m, mode) for m in motif]]
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield ''.join(prod)


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

def sci_ticks(ax, axis, ticksize=7, pad=4):
    
    if axis == 'x':
        xl, xh = ax.set_xlim()
        minorTicks = np.log10(np.arange(1, 11))
        minorTicks = np.concatenate([minorTicks+i for i in np.arange((xl//1), (xh//1)+1)])
        minorTicks = minorTicks[(minorTicks >= xl) & (minorTicks <= xh)]
        ax.set_xticks(minorTicks, minor=True)

        majorTicks = np.arange((xl//1)+1, (xh//1)+1)
        ax.set_xticks(majorTicks, [f'$10^{int(i)}$' for i in majorTicks])
        ax.tick_params(axis='x', which='major', direction='in', width=0.3, length=2, labelsize=ticksize, pad=pad)
        ax.tick_params(axis='x', which='minor', direction='in', width=0.3, length=1)
        
    elif axis == 'y':
        yl, yh = ax.set_ylim()
        minorTicks = np.log10(np.arange(1, 11))
        minorTicks = np.concatenate([minorTicks+i for i in np.arange((yl//1), (yh//1)+1)])
        minorTicks = minorTicks[(minorTicks >= yl) & (minorTicks <= yh)]
        ax.set_yticks(minorTicks, minor=True)

        majorTicks = np.arange((yl//1)+1, (yh//1)+1)
        ax.set_yticks(majorTicks, [f'$10^{int(i)}$' for i in majorTicks])
        ax.tick_params(axis='y', which='major', direction='in', width=0.3, length=2, labelsize=ticksize, pad=pad)
        ax.tick_params(axis='y', which='minor', direction='in', width=0.3, length=1)
        
    return None

def showText(ax, text, x=.5, y=.5, fontsize=16, fontweight='normal', rotation='horizontal', va='center', ha='center', color='black'):
    xl, xh = ax.set_xlim()
    yl, yh = ax.set_ylim()
    ax.text(xl+(xh-xl)*x, yl+(yh-yl)*y, text, 
            fontsize=fontsize, fontweight=fontweight, va=va, ha=ha, rotation=rotation, color=color)
    return None

def readTXT(file):
    with open(file) as f:
        content = f.readlines()
    return [x.strip() for x in content]

def writeTXT(file_path, lines, mode='w'):
    with open(file_path, mode) as file:
        for line in lines:
            file.write(f"{line}\n")
    return None
