import torch
from modify_traj_utils import modify_func
traj_pred = torch.randn([500,100,5,2]).cuda()
cls_pred = torch.randn([500,100]).cuda()
import time
start_time = time.time()
out_dict = modify_func(
                output = dict(
                    reg = [traj_p for traj_p in traj_pred.detach().unsqueeze(1)],
                    cls = [cls for cls in cls_pred.detach().unsqueeze(1)],
                ),
                num_mods_out=6
            )
traj_pred = torch.cat(out_dict['reg'],dim=0)
cls_pred = torch.cat(out_dict['cls'],dim=0)
end_time = time.time()
print("Time: ", end_time - start_time)