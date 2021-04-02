import csv
from typing import Sequence, Callable
import numpy
        
class RotationTransform():
    def __init__(self, seed: int=None):
        self.rng = numpy.random.RandomState()
        self.seed(seed)
        self._2pi = 2*numpy.pi

    def seed(self, s):
        self.rng.seed(s)

    def __call__(self, entry):
        d = self.rng.random() * self._2pi
        s, c = numpy.sin(d), numpy.cos(d)
        R = numpy.asarray([
            [c, -s],
            [s,  c]
        ])
        ref_a = entry["agent"]
        assert(len(ref_a) == 4)
        ref_a = (R @ numpy.array(ref_a).reshape(-1, 2, 1)).reshape(-1)
        return dict(
            neighbors = (R @ numpy.array(entry["neighbors"]).reshape(-1, 2, 1)).reshape(-1, 4),
            agent = ref_a,
            output = R @ entry["output"]
        )

class FlipTransform():
    def __init__(self, seed: int=None):
        self.rng = numpy.random.RandomState()
        self.seed(seed)

    def seed(self, s):
        self.rng.seed(s)

    def __call__(self, entry):
        ref_a, a, neighbors = entry["agent"], entry["output"], entry["neighbors"]
        flip_lr = self.rng.random() > 0.5
        flip_ud = self.rng.random() > 0.5
        if flip_lr and flip_ud:
            a = (-a[0], -a[1])
            neighbors = [(0., 0., 0., 0.)]
            for n in entry["neighbors"][1:]:
                neighbors.append((
                    -n[0], -n[1], -n[2], -n[3]
                ))
            ref_a = (-ref_a[0], -ref_a[1], -ref_a[2], -ref_a[3])
        elif flip_lr:
            a = (-a[0], a[1])
            neighbors = [(0., 0., 0., 0.)]
            for n in entry["neighbors"][1:]:
                neighbors.append((
                    -n[0], n[1], -n[2], n[3]
                ))
            ref_a = (-ref_a[0], ref_a[1], -ref_a[2], ref_a[3])
        elif flip_ud:
            a = (a[0], -a[1])
            neighbors = [(0., 0., 0., 0.)]
            for n in entry["neighbors"][1:]:
                neighbors.append((
                    n[0], -n[1], n[2], -n[3]
                ))
            ref_a = (ref_a[0], -ref_a[1], ref_a[2], -ref_a[3])
        return dict(
            neighbors = neighbors,
            agent = ref_a,
            output = a
        )


def as_pytorch_dataset(interpolation=False, **kwargs):
    import torch

    class Base():
        def __init__(self,
            data_files: Sequence[str],
            agent_ids: Sequence[int] or Sequence[Sequence[int]] = None,
            transform: Sequence[Callable[[dict], dict]] = None,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            seed: int = None
        ):
            self.transform = transform
            self.data = []
            self.agent_ids = []
            for i, f in enumerate(data_files):
                data = {}
                with open(f) as ff:
                    for row in csv.DictReader(ff):
                        idx = int(row["id"])
                        if idx not in data: data[idx] = dict()
                        neighbors = row["neighbors"].split(",") if row["neighbors"] else None
                        data[idx][float(row["t"])] = dict(
                            x=float(row["x"]), y=float(row["y"]),
                            vx=float(row["vx"]), vy=float(row["vy"]),
                            vx_=float(row["vx_"]), vy_=float(row["vy_"]),
                            gx=float(row["gx"]), gy=float(row["gy"]),
                            gvx=float(row["gvx"]), gvy=float(row["gvy"]),
                            neighbors=list(map(int, neighbors)) if neighbors else []
                        )
                if data:
                    for idx, frames in data.items():
                        frames["time"] = sorted(list(frames.keys()))
                    if not agent_ids:
                        cand_idx = list(data.keys())
                    elif hasattr(agent_ids[0], "__len__"):
                        if agent_ids[i]:
                            cand_idx = [idx for idx in agent_ids[i] if idx in data]
                        else:
                            cand_idx = list(data.keys())
                    else:
                        cand_idx = [idx for idx in agent_ids if idx in data]
                    if cand_idx:
                        self.data.append(data)
                        self.agent_ids.append(cand_idx)
            self.to(device)
            self.rng = numpy.random.RandomState()
            self.seed(seed)

        def batch_collator(self, batch):
            neighbors = [torch.tensor(entry["neighbors"], device=self.device, dtype=torch.float32) for entry in batch]
            agent = torch.tensor([entry["agent"] for entry in batch], device=self.device, dtype=torch.float32)
            output = torch.tensor([entry["output"] for entry in batch], device=self.device, dtype=torch.float32)
            return neighbors, agent, output
        
        def to(self, device):
            self.device = device

        def seed(self, seed: int = None):
            self.rng.seed(seed)

    if interpolation:
        class IterableDataset(Base, torch.utils.data.IterableDataset):
            def __init__(self, neighbor_radius=None, **kwargs):
                Base.__init__(self, **kwargs)
                if neighbor_radius:
                    self.neighbor_radius2 = neighbor_radius*neighbor_radius
                else:
                    self.neighbor_radius2 = 0
                self.neighbor_radius = neighbor_radius
                self._n_frames = 0
                for frames, ids in zip(self.data, self.agent_ids):
                    for i in ids:
                        self._n_frames += len(frames[i]) - 1

            def __len__(self) -> int:
                return self._n_frames

            def __iter__(self):
                self._iter_counter = 0
                return self
            
            def __next__(self):
                if self._iter_counter >= self._n_frames:
                    raise StopIteration

                if len(self.data) > 1:
                    env = self.rng.randint(0, len(self.data))
                else:
                    env = 0
                idx = self.rng.randint(0, len(self.agent_ids[env]))
                data = self.data[env]
                frames = data[idx]
                tid = self.rng.randint(0, len(frames["time"])-1) # skip last frame
                frac = self.rng.random()
                
                fid0 = frames["time"][tid]
                fid1 = frames["time"][tid+1]

                dt = fid1 - fid0
                f0 = frames[fid0]
                f1 = frames[fid1]

                if frac == 0:
                    n = [[0., 0., 0., 0.]]
                    for nid in f0["neighbors"]:
                        f = data[nid][fid0]
                        n.append([
                            f["x"]  - f0["x"], f["y"]  - f0["y"],
                            f["vx"] - f0["vx"], f["vy"] - f0["vy"]
                        ])
                    vx, vy = f0["vx"], f0["vy"]
                else:
                    _frac = 1-frac
                    x   = _frac*f0["x"] + frac*f1["x"]
                    y   = _frac*f0["y"] + frac*f1["y"]
                    vx  = _frac*f0["vx"] + frac*f1["vx"]
                    vy  = _frac*f0["vy"] + frac*f1["vy"]
                    vx_ = _frac*f0["vx_"] + frac*f1["vx_"]
                    vy_ = _frac*f0["vy_"] + frac*f1["vy_"]
                    
                    n = [[0., 0., 0., 0.]]
                    neighbors = set(f0["neighbors"]+f1["neighbors"])
                    for nid in neighbors:
                        frames = data[nid]
                        if fid0 in frames and fid1 in frames:
                            nf0 = frames[fid0]
                            nf1 = frames[fid1]
                            dx = _frac*nf0["x"] + frac*nf1["x"] - x
                            dy = _frac*nf0["y"] + frac*nf1["y"] - y
                            if self.neighbor_radius2 and dx*dx+dy*dy > self.neighbor_radius2:
                                continue
                            nvx = _frac*nf0["vx"] + frac*nf1["vx"]
                            nvy = _frac*nf0["vy"] + frac*nf1["vy"]
                        elif fid0 in frames:
                            nf0 = frames[fid0]
                            dx = nf0["x"] + dt*frac*nf0["gvx"] - x
                            dy = nf0["y"] + dt*frac*nf0["gvy"] - y
                            if self.neighbor_radius2 and dx*dx+dy*dy > self.neighbor_radius2:
                                continue
                            nvx = _frac*nf0["vx"] + frac*nf0["gvx"]
                            nvy = _frac*nf0["vy"] + frac*nf0["gvy"]
                        else: # fid1 in frames
                            nf1 = frames[fid1]
                            dx = nf1["x"] - dt*_frac*nf1["vx"] - x
                            dy = nf1["y"] - dt*_frac*nf1["vy"] - y
                            if self.neighbor_radius2 and dx*dx+dy*dy > self.neighbor_radius2:
                                continue
                            nvx, nvy = nf1["vx"], nf1["vy"]
                        
                        dvx, dvy = nvx - vx, nvy - vy
                        n.append((
                            dx, dy, dvx, dvy
                        ))
                dpx, dpy = f1["gx"]-x, f1["gy"]-y
                ref_a = (dpx, dpy, vx, vy)
                a = (vx_, vy_)
                
                self._iter_counter += 1
                entry = dict(
                    agent = ref_a,
                    neighbors = n,
                    output = a
                )
                if self.transform:
                    for trans in self.transform:
                        entry = trans(entry)
                return entry
        return IterableDataset(**kwargs)
    else:
        class Dataset(Base, torch.utils.data.Dataset):
            def __init__(self, neighbor_radius=None, **kwargs):
                Base.__init__(self, **kwargs)

                agent, output, neighbors = [], [], []
                for data, valid_agents in zip(self.data, self.agent_ids):
                    for aid, frames in data.items():
                        if aid not in valid_agents: continue
                        for i, t in enumerate(frames["time"]):
                            f = frames[t]
                            # skip last frame
                            if f["x"] == f["gx"] and f["y"] == f["gy"]:
                                continue
                            if i == len(frames["time"]) - 1:
                                dt = t - frames["time"][i-1]
                            else:
                                dt = frames["time"][i+1] - t
                            
                            n = [[0., 0., 0., 0.]]
                            for nid in f["neighbors"]:
                                n.append((
                                    data[nid][t]["x"]  - f["x"], data[nid][t]["y"]  - f["y"],
                                    data[nid][t]["vx"] - f["vx"], data[nid][t]["vy"] - f["vy"]
                                ))
                            a = (f["vx_"], f["vy_"])
                            vx, vy = f["vx"], f["vy"]
                            dpx, dpy = f["gx"]-f["x"], f["gy"]-f["y"] 
                            ref_a = (dpx, dpy, vx, vy)
                            
                            output.append(a)
                            agent.append(ref_a)
                            neighbors.append(n)

                self.agent_data = numpy.asarray(agent)
                self.neighbors_data = [numpy.array(n) for n in neighbors]
                self.output_data = numpy.asarray(output)


            def __len__(self) -> int:
                return len(self.agent_data)
            
            def __getitem__(self, idx: (Sequence[int], torch.Tensor)):
                if torch.is_tensor(idx):
                    idx = idx.cpu().tolist()
                entry = dict(
                    agent = self.agent_data[idx],
                    neighbors = self.neighbors_data[idx],
                    output = self.output_data[idx]
                )
                if self.transform:
                    for trans in self.transform:
                        entry = trans(entry)
                return entry
        return Dataset(**kwargs)
