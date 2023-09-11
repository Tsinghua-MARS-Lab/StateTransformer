import torch
def obtain_valid_index(mask, threshold):
    """
    Obtain valid indices where the sum along the trajectory is >= 2.
    """
    sums = mask.sum(dim=1)
    valid_indices = (sums >= int(threshold)).squeeze().nonzero(as_tuple=True)[0]
    return valid_indices

def get_range_indices(tensor):
        # Find the indices of the first and last occurrences of 1s along the rows for each column
    min_indices = (tensor == 1).int().argmax(dim=0)
    max_indices = (tensor == 1).int().flip(dims=[0]).argmax(dim=0)
    max_indices = tensor.size(0) - 1 - max_indices

    # Handle the case where there are no 1s in the column
    no_ones = (tensor.sum(dim=0) == 0)
    min_indices[no_ones] = -1
    max_indices[no_ones] = -1

    # Stack the indices together to form the pairs
    result = torch.stack([min_indices, max_indices], dim=1)

    return result

def interpolate_with_exp_interval(traj, mask):
    mask = mask.int()
    valid_index = obtain_valid_index(mask, 5)
    mask = mask[valid_index]
    traj = traj[valid_index]
    x_dist = torch.tensor([79, 39, 19, 9, 4]).float().to(traj.device).unsqueeze(0).unsqueeze(2)  # (1, l, 1)
    b, l, d = traj.shape
    # print(mask[:,:,0])
    valid_pts = mask[:, :, 0].nonzero()  # Returns indices where mask is non-zero
    # print(valid_pts)
    max_pairs = get_range_indices(mask[:,:,0].t())
    # print(max_pairs)
    first_valid_idx = max_pairs[:,0]
    last_valid_idx = max_pairs[:,1]

    # print(first_valid_idx)
    # print(last_valid_idx)
    # print("sadlkkfnwoeu")
    start = torch.gather(traj, 1, first_valid_idx.unsqueeze(1).unsqueeze(2).expand(b, 1, d))
    end = torch.gather(traj, 1, last_valid_idx.unsqueeze(1).unsqueeze(2).expand(b, 1, d))
    delta_x = (x_dist[:, last_valid_idx, :] - x_dist[:, first_valid_idx, :]).squeeze(0).unsqueeze(-1)
    # print(start.shape)
    # print(end.shape)
    # print(delta_x.shape)
    # print(x_dist[0, first_valid_idx, :].unsqueeze(-1).shape)
    _k = (end - start) / delta_x
    _b = start - _k * x_dist[0, first_valid_idx, :].unsqueeze(-1)
    # print(_k.shape)
    # print(_b.shape)
    # print("dslivbuquw9oefhdjnkm")
    interpolated_traj = _k * x_dist + _b
    # print(interpolated_traj)
    # print(interpolated_traj.shape)

    # Use mask to replace known points from the original traj
    mask_expanded = mask.expand(b, l, d)
    interpolated_traj = torch.where(mask_expanded.bool(), traj, interpolated_traj)

    return interpolated_traj
x_lst = list()
y_lst = list()
z_lst = list()
for i in range(1000):
    x = torch.load(f'/localdata_ssd/waymo_1/diffusion_dataset/test/future_key_points_hidden_state_{i}.pth')
    y = torch.load(f'/localdata_ssd/waymo_1/diffusion_dataset/test/future_key_points_{i}.pth')
    z = torch.load(f'/localdata_ssd/waymo_1/diffusion_dataset/test/future_key_points_gt_mask_{i}.pth')
    x_lst.append(x)
    y_lst.append(y)
    z_lst.append(z)
x = torch.cat(x_lst,dim=0)
y = torch.cat(y_lst,dim=0)
z = torch.cat(z_lst,dim=0)

print(y.max(dim=0))
print(y.min(dim=0))
print(y[18552])
print(y.shape)
y = interpolate_with_exp_interval(y,z)

print(y.max(dim=0))
print(y.min(dim=0))
print(y.shape)