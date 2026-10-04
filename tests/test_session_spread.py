"""Session collage: neurons of each area spread over all its sessions, E/I identity preserved, no neuron used twice."""
import numpy as np, torch
from infopath.config import config_vahid
from infopath.model_loader import FullModel
from infopath.session_stitching import build_network
import pandas as pd

opt = config_vahid(); opt.n_units = 1000; opt.device = torch.device("cpu")
model = FullModel(opt)
np.random.seed(0)
ni, fr, sess, area = build_network(model.rsnn, opt.datapath, opt.areas, False, 0.0, spread_sessions=True)
cl = pd.read_csv(opt.datapath + "/cluster_information")
exc = model.rsnn.excitatory_index.numpy(); aidx = model.rsnn.area_index.numpy()
for a, an in enumerate(opt.areas):
    print(an, {k[:15]: int(v) for k, v in pd.Series(sess[aidx == a]).value_counts().sort_index().items()})
n_sessions = {an: len(set(sess[aidx == a])) for a, an in enumerate(opt.areas)}
assert n_sessions == {"ALM": 6, "AC": 5}, n_sessions
keys = list(zip(sess, ni.astype(int)))
assert len(set(keys)) == len(keys), "a recorded neuron was used twice"
for i in range(1000):                      # E/I label and area of the model unit equal the recorded neuron's
    row = cl[(cl.session == sess[i]) & (cl.cluster_index == int(ni[i]))].iloc[0]
    assert bool(row.excitatory) == bool(exc[i]) and row.area == opt.areas[aidx[i]]
print("OK: collage over", n_sessions, "sessions; unique neurons; E/I and area labels preserved")
