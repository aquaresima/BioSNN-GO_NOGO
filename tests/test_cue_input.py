"""Cue channel: a pulse at cue_time only on the cue channel, area routing per channel, all inputs excitatory."""
import torch
from infopath.config import config_vahid
from infopath.model_loader import FullModel

opt = config_vahid(); opt.n_units = 200; opt.device = torch.device("cpu"); opt.batch_size = 4
torch.manual_seed(0)
model = FullModel(opt)
stims = torch.tensor([0, 1, 0, 1]).float()
x = model.input_spikes(stims)                       # (T, batch, channels)
dt = opt.dt / 1000.0
assert x.shape[2] == 4, x.shape
base = x[0, 0, 0].item()
on = (x[:, :, 3] > base * 1.5).any(1)               # time steps where the cue channel is raised
idx = torch.where(on)[0]
t_start = idx[0].item() * dt + opt.start
print("cue pulse starts at %.3f s, lasts %d steps (%.2f s)" % (t_start, len(idx), len(idx) * dt))
assert abs(t_start - (opt.cue_time + opt.thalamic_delay)) < 2 * dt and abs(len(idx) * dt - opt.cue_duration) < 2 * dt
assert torch.equal(x[:, 0, 3], x[:, 1, 3]), "cue must be identical for GO and NO-GO"
sound = x[:, :, :3]
assert (x[(idx[0]):, :, :3] <= sound.max()).all()
# GO / NO-GO channels still differ by stimulus, GO channel raised only on GO trials
go_raised = (x[:, :, 1] > base * 1.5).any(0); nogo_raised = (x[:, :, 2] > base * 1.5).any(0)
assert go_raised.tolist() == [False, True, False, True] and nogo_raised.tolist() == [True, False, True, False]
# pre-stimulus copy (0.1 s) must not fail and must not contain a cue
pre = model.input_spikes_pre(torch.zeros(4).long())
assert pre.shape[2] == 4 and (pre[:, :, 3] <= base * 1.5).all()
# area routing and sign
w = model.rsnn._w_in.detach(); area = model.rsnn.area_index
names = opt.areas
alm, ac = area == names.index("ALM"), area == names.index("AC")
assert (w[alm][:, :3] == 0).all() and (w[ac][:, :3] != 0).any(), "sound channels must reach AC only"
assert (w[alm][:, 3] != 0).any() and (w[ac][:, 3] != 0).any(), "cue channel must reach both areas"
assert model.rsnn.n_exc_inp == 4, model.rsnn.n_exc_inp
print("OK: cue timing, GO/NO-GO/cue channels, area routing, all 4 input channels excitatory (n_exc_inp=%d)" % model.rsnn.n_exc_inp)
