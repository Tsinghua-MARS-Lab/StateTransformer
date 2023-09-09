# optimized version for traj_modify.py
import torch
import einops
import math
import time
INIT_THRESHOLD = 0.5


    
    

def farthest_point_sample_PointCloud(xy, npoint):
    """
    Input:
        xy: pointcloud data, [B, N, 2]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xy.device
    B, N, C = xy.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.zeros((B,), dtype=torch.long).to(device) # modified so that the zeroth point with max confidence is always sampled.
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xy[batch_indices, farthest, :].view(B, 1, 2)
        dist = torch.sum((xy - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def modify_func(output:dict,nms_method='fde',num_mods_out:int = 36, init_threshold: float = INIT_THRESHOLD, nms_or_fps = 'nms', EM_Iter = 50, org_sigma = 1e-1, init_sigma = 1e-1,):
    print("We are now using modifyTraj function defined in traj_modify_from_MultiPathPP.py.")
    """Modify the output trajectory using NMS.

    Args:
        output (dict): with two keys, 'cls' and 'reg'.
            'reg' is the traj predicted. It's a list of length #scene,
                each item is a tensor of shape agent * num_mods * pred_length * 2 (2-D)
            'cls' is the score given by the model for each traj predicted. It's a list of length #scene,
                each item is a tensor of shape agent * num_mods
    Returns:
        output (dict): with two keys 'cls' and 'reg'.
            'reg' is the traj predicted. It's a list of length #scene,
                each item is a tensor of shape agent * num_mods_out * pred_length * 2 (2-D)
            'cls' is the score given by the model for each traj predicted. It's a list of length #agent,
                each item is a tensor of shape agent * num_mods_out
            'sigma' is the squared uncertainty of the traj predicted. It's a list of length #scene,
                each item is a tensor of shape agent * num_mods_out * pred_length * 2 (2-D)
    Notice:
        We only use the first dim of num_scale, since this is what is done in class PostProcess defined in lanegcn.
    """
    # calc_value = calc_ades if nms_method == 'ade' else None
    # flag1 = time.time()
    # print("Currently using {} sampling method to initialize the output of EM algorithm. Default is fde.".format(nms_method))
    assert nms_method == "fde", 'Not implemented'
        
    

    
    ireg_lst = output['reg']
    icls_lst = output['cls']
    oreg_lst_ = list()
    ocls_lst_ = list()
    agent_nums = [x.shape[0] for x in ireg_lst]

    ireg_tensor = torch.cat(ireg_lst, dim=0)
    icls_tensor = torch.cat(icls_lst, dim=0)
    ireg_finalpoint_tensor = ireg_tensor[...,-1,:] # batchsize * num_mods_in * 2
    sampled_idx = farthest_point_sample_PointCloud(ireg_finalpoint_tensor, num_mods_out) # batchsize * num_mods_out
    # oreg_tensor_[b,i] = ireg_tensor[b, sampled_idx[b,i], :]
    oreg_tensor_ = torch.gather(ireg_tensor, 1, einops.repeat(sampled_idx,'b n -> b n l d', l = ireg_tensor.shape[2], d = ireg_tensor.shape[3]))
    ocls_tensor_ = torch.gather(icls_tensor, 1, sampled_idx)
    oreg_lst_ = torch.split(oreg_tensor_, agent_nums, dim=0)
    ocls_lst_ = torch.split(ocls_tensor_, agent_nums, dim=0)
    # flag2 = time.time()
    # oreg_lst_: list of length #batch, each element is a tensor of shape 1 * num_mods_out * pred_length * 2
    # ocls_lst_: list of length #batch, each element is a tensor of shape 1 * num_mods_out
    
    
    
    refined_oreg_lst_, refined_ocls_lst_, sigma_lst_ = run_EM_algorithm(ireg_lst, icls_lst, oreg_lst_, ocls_lst_, EM_Iter, org_sigma, init_sigma)
    # flag3 = time.time()
    # print("fps uses:", flag2 - flag1)
    # print("fps part 1 uses:", flag1_5 - flag1)
    # print("fps part 2 uses:", flag2 - flag1_5)
    # print("EM uses:", flag3 - flag2)
    return dict(
        reg = refined_oreg_lst_,
        cls = refined_ocls_lst_,
        sigma = sigma_lst_,
    )
    
    
    # now we use the oreg_lst_, ocls_lst_ to initialize a q, mu, sigma.

def calculate_N_Gaussian_with_clip(x_minus_mu, sigma):

    # calculate 1 / sqrt(2*pi*sigma)
    log_first_part = - 1/2 * torch.log(2. * math.pi * (sigma+1e-5))
    
    # calculate exp(-(x_minus_mu)**2 / (2*sigma))
    log_second_part = -(x_minus_mu)**2 / (2. * (sigma+1e-5))

    # the result for each dimension is first_part * second_part
    result = log_first_part + log_second_part

    # For a multivariate distribution, we multiply the result from each dimension.
    # As we want to do this operation for the last 2*length dimensions, we can reshape the result tensor 
    # and then use torch.prod() to multiply along these dimensions.
    shape = result.shape
    reshaped_result = result.reshape(shape[:-2] + (-1,))
    log_reshaped_result = reshaped_result
    summed_log_reshaped_result = torch.sum(log_reshaped_result,dim=-1)

    # That is, we let summed_log_reshaped_result[i,:] = summed_log_reshaped_result[i,:] - max_j summed_log_reshaped_result[i,j]
    # Why this is correct? since we only use this to calculate p(h|x,\bar \Phi) in which we use N / sum_{k=1}^M N, so for a same i, it is correct for N[i] to be divided by a same value.
    # Calculate maxima along the second dimension (j)
    max_vals, _ = torch.max(summed_log_reshaped_result, dim=-1, keepdim=True)

    # Subtract maxima from corresponding rows
    summed_log_reshaped_result = summed_log_reshaped_result - max_vals


    return torch.exp(summed_log_reshaped_result)



def _compute_EM_algorithm(targs):
    org_q,org_mu,org_sigma,current_q,current_mu,current_sigma,args_dict = targs
    em_iter = args_dict['em_iter']
    org_q = org_q
    org_mu = org_mu
    org_sigma = org_sigma
    current_q = current_q
    current_mu = current_mu
    current_sigma = current_sigma
    in_mods = org_q.shape[1]
    out_mods = current_q.shape[1]
    
    for _ in range(em_iter):
        # print("One iter!")
        # p(h | mu_i ; \bar \Phi) # #agent * out_mods
        delta_mu = org_mu.unsqueeze(2) - current_mu.unsqueeze(1) # agent * in_mods * out_mods * length * 2
        N_Gaussian_value = calculate_N_Gaussian_with_clip(delta_mu, current_sigma.unsqueeze(1).repeat(1,in_mods,1,1,1)) # agent * in_mods * out_mods (* length * 2)
        q_h_N_deltamu_sigma = current_q.unsqueeze(1) * N_Gaussian_value # agent * in_mods * out_mods
        q_h_N_deltamu_sigma = q_h_N_deltamu_sigma / torch.sum(q_h_N_deltamu_sigma, dim = -1 ,keepdim=True) # agent * in_mods * out_mods

        new_q = torch.einsum('ai,aio->ao',org_q,q_h_N_deltamu_sigma)
        new_mu = torch.einsum('aio,aild->aold',org_q.unsqueeze(2) * q_h_N_deltamu_sigma, org_mu)/(new_q+1e-10).unsqueeze(-1).unsqueeze(-1)
        delta_mu_prime = org_mu.unsqueeze(2) - new_mu.unsqueeze(1) # agent * in_mods * out_mods * length * 2
        new_sigma = torch.einsum('aio,aiold->aold',org_q.unsqueeze(2) * q_h_N_deltamu_sigma, org_sigma.unsqueeze(2).repeat(1,1,out_mods,1,1) + delta_mu_prime**2)/(new_q+1e-10).unsqueeze(-1).unsqueeze(-1)

        current_q = new_q
        current_mu = new_mu
        current_sigma = new_sigma
    
    current_q, indices = torch.sort(current_q, descending=True, dim=1)  # shape: (agent, out_mods)

    # Use indices to rearrange current_mu and current_sigma
    current_mu = torch.gather(current_mu, 1, indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, current_mu.shape[-2], 2))  # shape: (agent, out_mods, length, 2)
    current_sigma = torch.gather(current_sigma, 1, indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, current_mu.shape[-2], 2))  # shape: (agent, out_mods, length, 2)
    return (current_q, current_mu, current_sigma)

def run_EM_algorithm(ireg_lst, icls_lst, oreg_lst, ocls_lst, EM_Iter, org_sigma, init_sigma):
    print("Now we run for {} iterations of EM algorithm.".format(EM_Iter))
    # ireg_lst: list of length #scene, each element is a tensor of shape #agent_in_current_scene * in_mods * #pred_length * 2
    # icls_lst: list of length #scene,                          of shape #agent_in_current_scene * in_mods
    # oreg_lst:                #scene                                    #agent_in_current_scene * out_mods * #pred_length * 2
    # ocls_lst:                #scene                                    #agent_in_current_scene * out_mods
    
    
    
    # This is the EM algorithm mentioned in the paper of MultiPath++.
    # we use the trajectories obtained by fps as an initialization.
    agent_nums = [x.shape[0] for x in ireg_lst]

    
    
    org_q = torch.cat(icls_lst,dim=0)   # #agent * in_mods
    org_q_sum = torch.sum(org_q, dim=-1, keepdim = True)
    org_q = org_q / org_q_sum

    org_mu = torch.cat(ireg_lst,dim=0)  # #agent * in_mods * #pred_length * 2
    org_mu_mean = torch.mean(org_mu,dim=1,keepdim=False).unsqueeze(1) # #agent * 1 * #pred_length * 2
    org_mu = org_mu - org_mu_mean
    org_sigma = torch.ones_like(org_mu) * float(org_sigma) # #agent * in_mods * #pred_length * 2, init with value close to 0.
    
    current_q = torch.cat(ocls_lst,dim=0)  # #agent * out_mods
    current_q_sum = torch.sum(current_q, dim=-1,keepdim=True)
    current_q = current_q / current_q_sum
    current_mu = torch.cat(oreg_lst,dim=0) # #agent * out_mods * #pred_length * 2
    current_mu = current_mu - org_mu_mean
    current_sigma = torch.ones_like(current_mu) * float(init_sigma) # #agent * out_mods * #pred_length * 2, init with value close to 0.

    in_mods = org_q.shape[1]
    out_mods = current_q.shape[1]

    org_q = 0*org_q + 1/in_mods
    current_q = 0*current_q + 1/out_mods
    
    CHUNKSIZE = 999999
    chunk_idx = ([CHUNKSIZE] * (len(org_q) // CHUNKSIZE) + [len(org_q) % CHUNKSIZE]) if len(org_q) % CHUNKSIZE != 0 else [CHUNKSIZE] * (len(org_q) // CHUNKSIZE)
    chunk_list_org_q = torch.split(org_q, chunk_idx, dim=0)
    chunk_list_org_mu = torch.split(org_mu, chunk_idx, dim=0)
    chunk_list_org_sigma = torch.split(org_sigma, chunk_idx, dim=0)
    chunk_list_current_q = torch.split(current_q, chunk_idx, dim=0)
    chunk_list_current_mu = torch.split(current_mu, chunk_idx, dim=0)
    chunk_list_current_sigma = torch.split(current_sigma, chunk_idx, dim=0)
    arg_list = [dict(em_iter = EM_Iter) for _ in range(len(chunk_list_org_q))]
    # print("AAAAAAAAAAuasiwefojdiwuleuj")
    # with multiprocessing.Pool(processes=30) as pool:
    #     result_lst = pool.map(_compute_EM_algorithm, zip(chunk_list_org_q, chunk_list_org_mu, chunk_list_org_sigma, chunk_list_current_q, chunk_list_current_mu, chunk_list_current_sigma, arg_list))
    result_lst = list()
    for item in zip(chunk_list_org_q, chunk_list_org_mu, chunk_list_org_sigma, chunk_list_current_q, chunk_list_current_mu, chunk_list_current_sigma, arg_list):
        result_lst.append(_compute_EM_algorithm(item))
    new_q_lst, new_mu_lst, new_sigma_lst = zip(*result_lst)
    new_q = torch.cat(new_q_lst,dim=0)
    new_mu = torch.cat(new_mu_lst,dim=0)
    new_sigma = torch.cat(new_sigma_lst,dim=0)
    new_mu = new_mu + org_mu_mean
    
    new_q_lst = torch.split(new_q, agent_nums, dim=0)
    new_mu_lst = torch.split(new_mu, agent_nums, dim=0)
    new_sigma_lst = torch.split(new_sigma, agent_nums, dim=0)
    return new_mu_lst, new_q_lst, new_sigma_lst
    
    

    # def calculate_N_Gaussian_with_clip(x_minus_mu, sigma):

    #     # calculate 1 / sqrt(2*pi*sigma)
    #     log_first_part = - 1/2 * torch.log(2. * torch.pi * (sigma+1e-5))
        
    #     # calculate exp(-(x_minus_mu)**2 / (2*sigma))
    #     log_second_part = -(x_minus_mu)**2 / (2. * (sigma+1e-5))

    #     # the result for each dimension is first_part * second_part
    #     result = log_first_part + log_second_part

    #     # For a multivariate distribution, we multiply the result from each dimension.
    #     # As we want to do this operation for the last 2*length dimensions, we can reshape the result tensor 
    #     # and then use torch.prod() to multiply along these dimensions.
    #     shape = result.shape
    #     reshaped_result = result.reshape(shape[:-2] + (-1,))
    #     log_reshaped_result = reshaped_result
    #     summed_log_reshaped_result = torch.sum(log_reshaped_result,dim=-1)

    #     # That is, we let summed_log_reshaped_result[i,:] = summed_log_reshaped_result[i,:] - max_j summed_log_reshaped_result[i,j]
    #     # Why this is correct? since we only use this to calculate p(h|x,\bar \Phi) in which we use N / sum_{k=1}^M N, so for a same i, it is correct for N[i] to be divided by a same value.
    #     # Calculate maxima along the second dimension (j)
    #     max_vals, _ = torch.max(summed_log_reshaped_result, dim=-1, keepdim=True)

    #     # Subtract maxima from corresponding rows
    #     summed_log_reshaped_result = summed_log_reshaped_result - max_vals


    #     return torch.exp(summed_log_reshaped_result)
        
    
    
    # for _ in range(EM_ITERATIONS):
    #     # p(h | mu_i ; \bar \Phi) # #agent * out_mods
    #     delta_mu = org_mu.unsqueeze(2) - current_mu.unsqueeze(1) # agent * in_mods * out_mods * length * 2
    #     N_Gaussian_value = calculate_N_Gaussian_with_clip(delta_mu, current_sigma.unsqueeze(1).repeat(1,in_mods,1,1,1)) # agent * in_mods * out_mods (* length * 2)
    #     q_h_N_deltamu_sigma = current_q.unsqueeze(1) * N_Gaussian_value # agent * in_mods * out_mods
    #     q_h_N_deltamu_sigma = q_h_N_deltamu_sigma / torch.sum(q_h_N_deltamu_sigma, dim = -1 ,keepdim=True) # agent * in_mods * out_mods

    #     new_q = torch.einsum('ai,aio->ao',org_q,q_h_N_deltamu_sigma)
    #     new_mu = torch.einsum('aio,aild->aold',org_q.unsqueeze(2) * q_h_N_deltamu_sigma, org_mu)/(new_q+1e-10).unsqueeze(-1).unsqueeze(-1)
    #     delta_mu_prime = org_mu.unsqueeze(2) - new_mu.unsqueeze(1) # agent * in_mods * out_mods * length * 2
    #     new_sigma = torch.einsum('aio,aiold->aold',org_q.unsqueeze(2) * q_h_N_deltamu_sigma, org_sigma.unsqueeze(2).repeat(1,1,out_mods,1,1) + delta_mu_prime**2)/(new_q+1e-10).unsqueeze(-1).unsqueeze(-1)

    #     current_q = new_q
    #     current_mu = new_mu
    #     current_sigma = new_sigma
    
    # current_mu = current_mu + org_mu_mean
    # each_q_list = torch.split(current_q,agent_nums,dim=0)
    # each_mu_list = torch.split(current_mu,agent_nums,dim=0)
    # each_sigma_list = torch.split(current_sigma,agent_nums,dim=0)
    # return each_mu_list, each_q_list, each_sigma_list