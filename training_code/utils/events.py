import torch
import torch.nn as nn
import pdb

def calc_floor_ceil_delta(x): 
    x_fl = torch.floor(x + 1e-8)
    x_ce = torch.ceil(x - 1e-8)
    x_ce_fake = torch.floor(x) + 1

    dx_ce = x - x_fl
    dx_fl = x_ce_fake - x

    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]

def create_update(x, y, t, dt, p, vol_size):
    vol_mul = torch.where(p < 0,
                          x.new_ones(p.shape, dtype=torch.long) * vol_size[0] // 2,
                          x.new_zeros(p.shape, dtype=torch.long))

    inds = (vol_size[1]*vol_size[2]) * (t + vol_mul)\
         + (vol_size[2])*y\
         + x

    vals = dt

    return inds, vals
 
def create_batch_update_volume(x, dx, y, dy, t, dt, p, vol_size):
    vol_mul = torch.where(p < 0,
                          x.new_ones(p.shape, dtype=torch.long) * vol_size[1] // 2,
                          x.new_zeros(p.shape, dtype=torch.long))

    batch_inds = torch.arange(x.shape[0], dtype=torch.long, device=x.device)[:, None]
    batch_inds = batch_inds.repeat((1, x.shape[1]))
    batch_inds = torch.reshape(batch_inds, (-1,))

    dx = torch.reshape(dx, (-1,))
    dy = torch.reshape(dy, (-1,))
    dt = torch.reshape(dt, (-1,))
    x = torch.reshape(x, (-1,))
    y = torch.reshape(y, (-1,))
    t = torch.reshape(t, (-1,))
    vol_mul = torch.reshape(vol_mul, (-1,))

    inds = torch.stack((batch_inds, t+vol_mul, y, x), dim=0)

    vals = dx * dy * dt

    return inds, vals

def create_batch_update_image(x, dx, y, dy, p, image_size):
    batch_inds = torch.arange(x.shape[0], dtype=torch.long, device=x.device)[:, None]
    batch_inds = batch_inds.repeat((1, x.shape[1]))
    batch_inds = torch.reshape(batch_inds, (-1,))

    dx = torch.reshape(dx, (-1,))
    dy = torch.reshape(dy, (-1,))
    x = torch.reshape(x, (-1,))
    y = torch.reshape(y, (-1,))
    p = torch.reshape(p, (-1,))

    # Find all events outsize of the sensor
    keep_mask = (x >= 0) * (y >= 0) * (x < image_size[-2]) * (y < image_size[-1])

    inds = torch.stack((batch_inds, y, x), dim=0)

    vals = dx * dy

    return inds, vals, keep_mask

def gen_discretized_event_volume(events, vol_size):
    # volume is [t, x, y]
    # events are Nx4
    npts = events.shape[0]
    volume = events.new_zeros(vol_size)

    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]

    t_min = t.min()
    t_max = t.max()
    t_scaled = (t-t_min) * ((vol_size[0] // 2-1) / (t_max-t_min))

    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())
    
    inds_fl, vals_fl = create_update(x, y,
                                     ts_fl[0], ts_fl[1],
                                     events[:, 3],
                                     vol_size)
        
    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)

    inds_ce, vals_ce = create_update(x, y,
                                     ts_ce[0], ts_ce[1],
                                     events[:, 3],
                                     vol_size)
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)

    return volume


def gen_batch_discretized_event_volume(events, vol_size):
    # vol_size is [b, t, x, y]
    # events are BxNx4
    batch = events.shape[0]
    npts = events.shape[1]
    volume = events.new_zeros(vol_size)

    # Each is BxN
    x = events[..., 0]
    y = events[..., 1]
    t = events[..., 2]

    # Dim is now Bx1
    t_min = t.min(dim=1, keepdim=True)[0]
    t_max = t.max(dim=1, keepdim=True)[0]
    t_scaled = (t-t_min) * ((vol_size[1]//2 - 1) / (t_max-t_min))

    xs_fcd = calc_floor_ceil_delta(x)
    ys_fcd = calc_floor_ceil_delta(y)
    ts_fcd = calc_floor_ceil_delta(t_scaled)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                inds, vals = create_batch_update_volume(xs_fcd[i][0], xs_fcd[i][1],
                                                        ys_fcd[j][0], ys_fcd[j][1],
                                                        ts_fcd[k][0], ts_fcd[k][1],
                                                        events[...,3],
                                                        vol_size)

                volume = volume + volume.index_put(tuple(inds), vals, accumulate=True)

    return volume

def gen_batch_event_images(events, image_size):
    # image_size is [b, 1, x, y]
    # events are BxNx4
    outputs = {}

    batch = events.shape[0]
    npts = events.shape[1]

    # Each is BxN
    x = events[..., 0]
    y = events[..., 1]
    t = events[..., 2]
    p = events[..., 3]

    # Dim is now Bx1
    t_min = t.min(dim=1, keepdim=True)[0]
    t_max = t.max(dim=1, keepdim=True)[0]
    t_scaled = (t-t_min) / (t_max-t_min+1e-3)

    all_inds = []
    all_vals = []
    all_counts = []
    pos_inds = []
    pos_vals = []
    pos_counts = []
    neg_inds = []
    neg_vals = []
    neg_counts = []

    pos_mask = p.view(-1) > 0.
    neg_mask = p.view(-1) <= 0.

    xs_updates = calc_floor_ceil_delta(x)
    ys_updates = calc_floor_ceil_delta(y)

    for i in range(2):
        for j in range(2):
            ind, val, keep_mask = create_batch_update_image(xs_updates[i][0], xs_updates[i][1],
                                                 ys_updates[j][0], ys_updates[j][1],
                                                 p,
                                                 image_size)

            all_inds.append(ind[:,keep_mask])
            all_vals.append(val[keep_mask] * t_scaled.view(-1)[keep_mask])
            all_counts.append(torch.ones_like(val[keep_mask]))

            pos_inds.append(ind[ :, pos_mask*keep_mask ])
            pos_vals.append(val[ pos_mask*keep_mask ] * t_scaled.view(-1)[ pos_mask*keep_mask ])
            pos_counts.append(torch.ones_like(pos_vals[-1]))

            neg_inds.append(ind[ :, neg_mask*keep_mask ])
            neg_vals.append(val[ neg_mask*keep_mask ] * t_scaled.view(-1)[ neg_mask*keep_mask ])
            neg_counts.append(torch.ones_like(neg_vals[-1]))

    all_inds = torch.cat(all_inds, dim=1)
    all_vals = torch.cat(all_vals, dim=0)
    all_counts = torch.cat(all_counts, dim=0)

    pos_inds = torch.cat(pos_inds, dim=1)
    pos_vals = torch.cat(pos_vals, dim=0)
    pos_counts = torch.cat(pos_counts, dim=0)

    neg_inds = torch.cat(neg_inds, dim=1)
    neg_vals = torch.cat(neg_vals, dim=0)
    neg_counts = torch.cat(neg_counts, dim=0)

    tmp = events.new_zeros(image_size).squeeze()

    ntsi = tmp.index_put(tuple(neg_inds), neg_vals, accumulate=True)
    nci = tmp.index_put(tuple(neg_inds), neg_counts, accumulate=True)

    ptsi = tmp.index_put(tuple(pos_inds), pos_vals, accumulate=True)
    pci = tmp.index_put(tuple(pos_inds), pos_counts, accumulate=True)

    outputs["neg_timestamp_image"] = ntsi / ( nci + 1e-4 )
    outputs["pos_timestamp_image"] = ptsi / ( pci + 1e-4 )

    outputs = {k:v.view(image_size) for k,v in outputs.items()}

    return outputs
