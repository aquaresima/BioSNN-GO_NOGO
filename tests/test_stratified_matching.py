import torch, numpy as np
from infopath.losses import (trial_matching_loss, hard_trial_matching_loss, stratified_psth_loss,
                             data_stim_tensor, z_score_norm)
torch.manual_seed(0); np.random.seed(0)
T, Ns = 60, 20
pat = [torch.sin(torch.linspace(0, 6, T)) * 1.0, torch.cos(torch.linspace(0, 6, T)) * 1.0]  # GO / NO-GO time courses
n_tr = [40, 30]                       # data trials per session
stims_s = [torch.randint(0, 2, (n,)) for n in n_tr]
N = 2 * Ns
data = torch.full((T, max(n_tr), N), float("nan"))
for s in range(2):
    for k in range(n_tr[s]):
        data[:, k, s*Ns:(s+1)*Ns] = pat[int(stims_s[s][k])][:, None] + 0.3 * torch.randn(T, Ns)
idx = [torch.arange(N) // Ns == s for s in range(2)]
session_info = [[torch.zeros(n, dtype=torch.long) for n in n_tr], stims_s, [None, None], idx]
K = 80
m_stim = torch.randint(0, 2, (K,))
def model_act(inverted):
    out = torch.zeros(T, K, N)
    for k in range(K):
        g = int(m_stim[k]) ^ int(inverted)
        out[:, k] = pat[g][:, None] + 0.3 * torch.randn(T, N)
    return out
area = torch.arange(N) // Ns; exc = torch.ones(N, dtype=torch.bool)
dstim = data_stim_tensor(session_info, data.shape[1], N, data.device)
print('data_stim ok:', int((dstim[:40, :Ns] == -1).sum()) == 0 and int((dstim[30:, Ns:] != -1).sum()) == 0)
res = {}
for name, inv in (('correct', False), ('inverted', True)):
    m = model_act(inv)
    psth_old = ((lambda a, b, c: ((a - b) ** 2).mean())(*z_score_norm(data, m)[:2], None)).item()
    psth_new = stratified_psth_loss(data, m, dstim, m_stim).item()
    tm_old = np.mean([trial_matching_loss(None, data, m, session_info, None, None, hard_trial_matching_loss, area, exc, stims=None).item() for _ in range(5)])
    tm_new = np.mean([trial_matching_loss(None, data, m, session_info, None, None, hard_trial_matching_loss, area, exc, stims=m_stim).item() for _ in range(5)])
    res[name] = (psth_old, psth_new, tm_old, tm_new)
    print(f'{name:9s} psth(old) {psth_old:.4f}  psth(stratified) {psth_new:.4f} | trial-match(old) {tm_old:.4f}  trial-match(stratified) {tm_new:.4f}')
print('old losses cannot tell correct from inverted:', abs(res['correct'][0]-res['inverted'][0]) < 0.1*res['correct'][0]+0.05, abs(res['correct'][2]-res['inverted'][2]) < 0.3*res['correct'][2]+0.05)
print('stratified losses separate them (inverted > 3x correct):', res['inverted'][1] > 3*res['correct'][1], res['inverted'][3] > 3*res['correct'][3])
