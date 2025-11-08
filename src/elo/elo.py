import math
from collections import defaultdict
H_ELO = 55.0
PTS_PER_ELO = 0.085
def exp_win_prob(rh, ra, H=H_ELO): return 1/(1+10**(-(((rh+H)-ra)/400)))
def margin_scaler(m, d): return min(math.log(m+1)* (2.2/(2.2+0.001*abs(d))), 1.8)
def k_factor(n, base=22.0): return base/(1+max(n,0))**0.5
class Elo:
    def __init__(self): self.R=defaultdict(lambda:1500.0); self.N=defaultdict(int)
    def predict(self, home, away, neutral=False):
        H=0.0 if neutral else H_ELO; diff=(self.R[home]+H)-self.R[away]
        return exp_win_prob(self.R[home], self.R[away], H), diff*PTS_PER_ELO
    def update(self, home, away, ph, pa, neutral=False):
        H=0.0 if neutral else H_ELO; Rh,Ra=self.R[home],self.R[away]
        pre=((Rh+H)-Ra); p=1/(1+10**(-pre/400)); s=1.0 if ph>pa else 0.0
        g=margin_scaler(abs(ph-pa), pre); Kh=k_factor(self.N[home]); Ka=k_factor(self.N[away])
        d=g*(s-p); self.R[home]=Rh+Kh*d; self.R[away]=Ra-Ka*d; self.N[home]+=1; self.N[away]+=1
