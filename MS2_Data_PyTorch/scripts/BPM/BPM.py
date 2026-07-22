import argparse
from BPM.util import *
import os
datapath = os.path.dirname(os.path.abspath(__file__)) + os.sep


# def _load_model():
#     Params = load_obj(datapath + "Params.pkl")
#     params_pwm = Params[0][:, 0][:36]
#     params_sp = Params[0][:, 0][36:]
#     PSAM35 = Params2PSAM(params_pwm[:18], 'dataframe')
#     PSAM10 = Params2PSAM(params_pwm[18:], 'dataframe')
#     dGSP = {15: params_sp[1]*4, 16: params_sp[1], 17: 0, 18: params_sp[2], 19: params_sp[2]*3}
#     c0, c1 = Params[2]**2
#     return PSAM35, PSAM10, dGSP, (c0, c1)

def _load_model():
    Params = load_obj(datapath + "Params_Con17.pkl")
    PSAM35 = Params['pwm35']
    PSAM10 = Params['pwm10']
    params_sp = Params['spacer energy']
    dGSP = {15: params_sp[16]*4, 16: params_sp[16], 17: 0, 18: params_sp[18], 19: params_sp[18]*3}
    cE0, cmin, cmax = (Params['cbound'], np.exp(Params['cmin']), np.exp(Params['cmax']))
    return PSAM35, PSAM10, dGSP, (cE0, cmin, cmax)

def score_m10(seq):
    if not seq in _score10:
        return 0.0
    return _score10[seq]

def score_m35(seq):
    if not seq in _score35:
        return 0.0
    return _score35[seq]

def score_spacer(seq):
    l = len(seq)
    return _dGSP[l]

def score_promoter(seq):
    m35 = seq[:6]
    m10 = seq[-6:]
    spacer = seq[6:-6]
    score = score_m35(m35) + score_spacer(spacer) + score_m10(m10)
    return score

def promoter_elements(seq):
    m35 = seq[:6]
    m10 = seq[-6:]
    spacer = seq[6:-6]
    return m35, spacer, m10

def score_promoter_elements(seq):
    m35, spacer, m10 = promoter_elements(seq)
    return score_m35(m35), score_spacer(spacer), score_m10(m10)

# def score2exp(dG):
#     return 1000 * (_C[0] * np.exp(1.623 * -dG)) + _C[1]

# def score2logexp(dG):
#     return np.log10(score2exp(dG))

# def predict(seq):
#     return score2logexp(score_promoter(seq))

def score2exp(dG):
    R = 0.001987
    T = 310
    beta = 1/R/T
    cE0, cmin, cmax = _C
    
    B_RNAP = np.exp(-beta * (dG+cE0))
    return (cmin + cmax*B_RNAP) / (1 + B_RNAP)

def score2logexp(dG):
    return np.log10(score2exp(dG))

def predict(seq):
    return score2logexp(score_promoter(seq))

# def promoter_iterator(seq):
#     sl = [16, 17, 18]
#     for l in sl:
#         for m in range(0, len(seq) - (12 + l) + 1):
#             yield (m, seq[m:m + (12 + l)])

# def promoter_argmax(seq):
#     maxi = -1
#     maxl = 0
#     maxs = -1

#     for i, s in promoter_iterator(seq):
#         x = predict(s)
#         if x > maxs:
#             maxs = x
#             maxi = i
#             maxl = len(s)

#     return maxi, maxl, maxs



_PSAM35, _PSAM10, _dGSP, _C = _load_model()
_score10 = Matrix2Score(_PSAM10)
_score35 = Matrix2Score(_PSAM35)




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Bacterial promoter strength prediction model prototype")
    parser.add_argument("command", help="command")
    parser.add_argument("input", help="Input sequence")
    
    args = parser.parse_args()


    cmd = args.command

    if cmd == "m10":
        print("-10:\t", score_m10(args.input))
    elif cmd == "m35":
        print("-35:\t", score_m35(args.input))
    elif cmd == "all":

        m35, spacer, m10 = score_promoter_elements(args.input)
        print("-35:\t", m35)
        print("spacer:\t", spacer)
        print("-10:\t", m10)
        print("total:\t", m35 + spacer + m10)
        print("exp:\t", score2exp(m35 + spacer + m10))
        print("logexp:\t", score2logexp(m35 + spacer + m10))
        
