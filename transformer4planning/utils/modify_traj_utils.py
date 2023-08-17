# optimized version for traj_modify.py
import torch
import einops
import multiprocessing
INIT_THRESHOLD = 0.5

def non_max_suppression(values, scores, threshold):
    # sort the trajs by their confidence score
    indices = list(range(len(scores)))
    indices.sort(key=lambda i: scores[i], reverse=True)

    keep = []
    while indices:
        i = indices.pop(0)
        keep.append(i)

        # find the trajs with overlapping area and remove them
        overlap_indices = []
        for j in indices:
            if values[i][j] < threshold:
                overlap_indices.append(j)

        indices = [i for i in indices if i not in overlap_indices]
    # print("Current num to keep is {}, current threshold is {}".format(len(keep),threshold))
    return keep
def calc_ades(ireg_ss):
    # ireg_s: tensor of shape agent * num_mods * pred_length * 2
    # returns: tensor of shape agent * num_mods * num_mods
    try:
        ireg_s = ireg_ss.detach().cuda()
        num_points = ireg_s.shape[1]
        # output[a] = torch.sqrt(torch.sum(torch.sum((ireg_s[a] - ireg_s[a].transpose(0,1))**2,dim=2),dim=1) / pred_length)
        output = torch.sqrt(torch.sum(torch.sum((einops.repeat(ireg_s,'a n p d -> a n w p d',w = ireg_s.shape[1]) \
                                            - einops.repeat(ireg_s,'a n p d -> a w n p d',w = ireg_s.shape[1]))**2,dim=4),dim=3) / num_points)
        return output.detach().cpu()
    except Exception as e:
        print("It seems that cuda is out of memory. Now we use cpu to calculate the ades. This may take a long time.")
        ireg_s = ireg_ss.detach().cpu()
        num_points = ireg_s.shape[1]
        # output[a] = torch.sqrt(torch.sum(torch.sum((ireg_s[a] - ireg_s[a].transpose(0,1))**2,dim=2),dim=1) / pred_length)
        output = torch.sqrt(torch.sum(torch.sum((einops.repeat(ireg_s,'a n p d -> a n w p d',w = ireg_s.shape[1]) \
                                            - einops.repeat(ireg_s,'a n p d -> a w n p d',w = ireg_s.shape[1]))**2,dim=4),dim=3) / num_points)
        return output.detach()

def calc_fdes(ireg_ss):
    try:
        ireg_s = ireg_ss[:,:,-1:,:].detach().cuda()
        num_points = ireg_s.shape[1]
        # output[a] = torch.sqrt(torch.sum(torch.sum((ireg_s[a] - ireg_s[a].transpose(0,1))**2,dim=2),dim=1) / pred_length)
        output = torch.sqrt(torch.sum(torch.sum((einops.repeat(ireg_s,'a n p d -> a n w p d',w = ireg_s.shape[1]) \
                                            - einops.repeat(ireg_s,'a n p d -> a w n p d',w = ireg_s.shape[1]))**2,dim=4),dim=3) / num_points)
        return output.detach().cpu()
    except Exception as e:
        print("It seems that cuda is out of memory. Now we use cpu to calculate the fdes. This may take a long time.")
        ireg_s = ireg_ss[:,:,-1:,:].detach().cpu()
        num_points = ireg_s.shape[1]
        # output[a] = torch.sqrt(torch.sum(torch.sum((ireg_s[a] - ireg_s[a].transpose(0,1))**2,dim=2),dim=1) / pred_length)
        output = torch.sqrt(torch.sum(torch.sum((einops.repeat(ireg_s,'a n p d -> a n w p d',w = ireg_s.shape[1]) \
                                            - einops.repeat(ireg_s,'a n p d -> a w n p d',w = ireg_s.shape[1]))**2,dim=4),dim=3) / num_points)
        return output.detach()


def compute_function_nms(targs):
    this_ades,ireg_s,icls_s,dct = targs
    init_threshold = dct['init_threshold']
    num_mods_out = dct['num_mods_out']
    oreg_lst = list()
    ocls_lst = list()
    for idx,(ireg,icls) in enumerate(zip(ireg_s,icls_s)):
        threshold = INIT_THRESHOLD
        keep_indices = non_max_suppression(this_ades[idx],icls,threshold)
        while(len(keep_indices)>int(1.4*num_mods_out)):
            threshold *= 1.1
            keep_indices = non_max_suppression(this_ades[idx],icls,threshold)
        while(len(keep_indices)<int(1.1*num_mods_out)):
            threshold/=1.1
            keep_indices = non_max_suppression(this_ades[idx],icls,threshold)
        
        # print("We are now using threshold of {} for current agent.".format(threshold))
        keep_indices = keep_indices[0:num_mods_out]
        oreg = ireg[keep_indices]
        ocls = icls[keep_indices]
        # print(oreg.shape)
        # print(ocls.shape)
        oreg_lst.append(oreg.unsqueeze(0))
        ocls_lst.append(ocls.unsqueeze(0))
    oreg_ = torch.cat(oreg_lst,dim=0)
    ocls_ = torch.cat(ocls_lst,dim=0)
    return (oreg_, ocls_)
def farthest_point_sampling(distances, scores, num_mods_out):
    # distances: N * N.
    # we first keep the 0th indice.
    # 
    # N = distances.shape[0]
    distances = distances.clone()[0:int(0.9*distances.shape[0]),0:int(0.9*distances.shape[0])]
    currenct_distances = distances.clone()[0]
    keep_indices = [0]
    for i in range(num_mods_out-1):
        next_index = torch.argmax(currenct_distances)
        keep_indices.append(next_index.item())
        currenct_distances = torch.min(currenct_distances,distances[next_index])
    return keep_indices
    
def compute_function_fps(targs):
    this_ades,ireg_s,icls_s,dct = targs
    init_threshold = dct['init_threshold']
    num_mods_out = dct['num_mods_out']
    oreg_lst = list()
    ocls_lst = list()
    for idx,(ireg,icls) in enumerate(zip(ireg_s,icls_s)):
        threshold = INIT_THRESHOLD
        keep_indices = farthest_point_sampling(this_ades[idx],icls,num_mods_out)
        # while(len(keep_indices)>int(1.4*num_mods_out)):
        #     threshold *= 1.1
        #     keep_indices = non_max_suppression(this_ades[idx],icls,threshold)
        # while(len(keep_indices)<int(1.1*num_mods_out)):
        #     threshold/=1.1
        #     keep_indices = non_max_suppression(this_ades[idx],icls,threshold)
        
        # print("We are now using threshold of {} for current agent.".format(threshold))
        keep_indices = keep_indices[0:num_mods_out]
        keep_indices = sorted(keep_indices)
        oreg = ireg[keep_indices]
        ocls = icls[keep_indices]
        # print(oreg.shape)
        # print(ocls.shape)
        oreg_lst.append(oreg.unsqueeze(0))
        ocls_lst.append(ocls.unsqueeze(0))
    oreg_ = torch.cat(oreg_lst,dim=0)
    ocls_ = torch.cat(ocls_lst,dim=0)
    return (oreg_, ocls_)
def random_return(output,num_mods_out):
    ireg_lst = [x.cpu() for x in output['reg']]
    icls_lst = [x.cpu() for x in output['cls']]
    oreg_lst_ = list()
    ocls_lst_ = list()
    for ireg,icls in zip(ireg_lst,icls_lst):
        index_tensor = torch.randint(0,ireg.shape[1],(num_mods_out,))
        oreg = ireg[:,index_tensor,:,:]
        ocls = icls[:,index_tensor]
        oreg_lst_.append(oreg)
        ocls_lst_.append(ocls)
    return dict(
        reg = oreg_lst_,
        cls = ocls_lst_,
    )
def modify_func(output:dict,nms_method='fde',num_mods_out:int = 36, init_threshold: float = INIT_THRESHOLD, nms_or_fps = 'nms', EM_Iter = 25, org_sigma = 1e-1, init_sigma = 1e-1,):
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
    print("Currently using {} sampling method to initialize the output of EM algorithm. Default is fde.".format(nms_method))
    if nms_method == "ade":
        calc_value = calc_ades
    elif nms_method == "fde":
        calc_value = calc_fdes
    elif nms_method == "random":
        return random_return(output,num_mods_out)
    else:
        assert False, "Not implemented."
        
    

    
    ireg_lst = output['reg']
    icls_lst = output['cls']
    oreg_lst_ = list()
    ocls_lst_ = list()
    agent_nums = [x.shape[0] for x in ireg_lst]
    big_ades = calc_value(torch.cat(ireg_lst,dim=0))
    each_ades_lst = torch.split(big_ades,agent_nums,dim=0)
    # each_ades_lst = [x.numpy() for x in each_ades_lst]
    ireg_lst = [x.cpu() for x in ireg_lst]
    icls_lst = [x.cpu() for x in icls_lst]
    dct_lst = [dict(
        init_threshold = init_threshold,
        num_mods_out = num_mods_out,
    )] * len(ireg_lst)
    # for outer_idx,(ireg_s,icls_s) in enumerate(zip(ireg_lst,icls_lst)):
    #     oreg_lst = list()
    #     ocls_lst = list()
    #     this_ades = each_ades_lst[outer_idx]
    #     for idx,(ireg,icls) in enumerate(zip(ireg_s,icls_s)):
    #         threshold = init_threshold
    #         keep_indices = non_max_suppression(this_ades[idx],icls,threshold)
    #         while(len(keep_indices)<int(1.1*num_mods_out)):
    #             threshold/=1.4
    #             keep_indices = non_max_suppression(this_ades[idx],icls,threshold)
    #         # print("We are now using threshold of {} for current agent.".format(threshold))
    #         keep_indices = keep_indices[0:num_mods_out]
    #         oreg = ireg[keep_indices]
    #         ocls = icls[keep_indices]
    #         # print(oreg.shape)
    #         # print(ocls.shape)
    #         oreg_lst.append(oreg.unsqueeze(0))
    #         ocls_lst.append(ocls.unsqueeze(0))
    #     oreg_ = torch.cat(oreg_lst,dim=0)
    #     ocls_ = torch.cat(ocls_lst,dim=0)
    #     # print(oreg_.shape)
    #     oreg_lst_.append(oreg_)
    #     ocls_lst_.append(ocls_)
    if nms_or_fps == 'nms':
        with multiprocessing.Pool(processes=15) as pool:
            output_lst = pool.map(compute_function_nms, zip(each_ades_lst,ireg_lst,icls_lst,dct_lst),chunksize = 2)
    elif nms_or_fps == 'fps':
        print("We are now using fps for sampling. Default: nms.")
        with multiprocessing.Pool(processes=15) as pool:
            output_lst = pool.map(compute_function_fps, zip(each_ades_lst,ireg_lst,icls_lst,dct_lst),chunksize = 2)
    oreg_lst_ = [x[0] for x in output_lst]
    ocls_lst_ = [x[1] for x in output_lst]
    
    refined_oreg_lst_, refined_ocls_lst_, sigma_lst_ = run_EM_algorithm(ireg_lst, icls_lst, oreg_lst_, ocls_lst_, EM_Iter, org_sigma, init_sigma)
    return dict(
        reg = refined_oreg_lst_,
        cls = refined_ocls_lst_,
        sigma = sigma_lst_,
    )
    
    
    # now we use the oreg_lst_, ocls_lst_ to initialize a q, mu, sigma.





def _compute_EM_algorithm(targs):
    org_q,org_mu,org_sigma,current_q,current_mu,current_sigma,args_dict = targs
    em_iter = args_dict['em_iter']
    org_q = org_q.cuda()
    org_mu = org_mu.cuda()
    org_sigma = org_sigma.cuda()
    current_q = current_q.cuda()
    current_mu = current_mu.cuda()
    current_sigma = current_sigma.cuda()
    in_mods = org_q.shape[1]
    out_mods = current_q.shape[1]
    def calculate_N_Gaussian_with_clip(x_minus_mu, sigma):

        # calculate 1 / sqrt(2*pi*sigma)
        log_first_part = - 1/2 * torch.log(2. * torch.pi * (sigma+1e-5))
        
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
    for _ in range(em_iter):
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
    return (current_q.cpu(), current_mu.cpu(), current_sigma.cpu())

def run_EM_algorithm(ireg_lst, icls_lst, oreg_lst, ocls_lst, EM_Iter, org_sigma, init_sigma):
    print("Now we run for {} iterations of EM algorithm.".format(EM_Iter))
    # torch.save(ireg_lst,'ireg_lst.pkl')
    # torch.save(icls_lst,'icls_lst.pkl')
    # torch.save(oreg_lst,'oreg_lst.pkl')
    # torch.save(ocls_lst,'ocls_lst.pkl')
    # assert False, 'debugging.'
    # This is the EM algorithm mentioned in the paper of MultiPath++.
    # we use the trajectories obtained by fps as an initialization.
    agent_nums = [x.shape[0] for x in ireg_lst]

    # ireg_lst: list of length #scene, each element is a tensor of shape #agent_in_current_scene * in_mods * #pred_length * 2
    # icls_lst: list of length #scene,                          of shape #agent_in_current_scene * in_mods
    # oreg_lst:                #scene                                    #agent_in_current_scene * out_mods * #pred_length * 2
    # ocls_lst:                #scene                                    #agent_in_current_scene * out_mods
    
    org_q = torch.cat(icls_lst,dim=0).cpu()   # #agent * in_mods
    org_q_sum = torch.sum(org_q, dim=-1, keepdim = True)
    org_q = org_q / org_q_sum

    org_mu = torch.cat(ireg_lst,dim=0).cpu()  # #agent * in_mods * #pred_length * 2
    org_mu_mean = torch.mean(org_mu,dim=1,keepdim=False).unsqueeze(1) # #agent * 1 * #pred_length * 2
    org_mu = org_mu - org_mu_mean
    org_sigma = torch.ones_like(org_mu) * float(org_sigma) # #agent * in_mods * #pred_length * 2, init with value close to 0.
    
    current_q = torch.cat(ocls_lst,dim=0).cpu()  # #agent * out_mods
    current_q_sum = torch.sum(current_q, dim=-1,keepdim=True)
    current_q = current_q / current_q_sum
    current_mu = torch.cat(oreg_lst,dim=0).cpu() # #agent * out_mods * #pred_length * 2
    current_mu = current_mu - org_mu_mean
    current_sigma = torch.ones_like(current_mu) * float(init_sigma) # #agent * out_mods * #pred_length * 2, init with value close to 0.

    in_mods = org_q.shape[1]
    out_mods = current_q.shape[1]

    org_q = 0*org_q + 1/in_mods
    current_q = 0*current_q + 1/out_mods
    
    CHUNKSIZE = 15
    chunk_idx = ([CHUNKSIZE] * (len(org_q) // CHUNKSIZE) + [len(org_q) % CHUNKSIZE]) if len(org_q) % CHUNKSIZE != 0 else [CHUNKSIZE] * (len(org_q) // CHUNKSIZE)
    chunk_list_org_q = torch.split(org_q.cpu(), chunk_idx, dim=0)
    chunk_list_org_mu = torch.split(org_mu.cpu(), chunk_idx, dim=0)
    chunk_list_org_sigma = torch.split(org_sigma.cpu(), chunk_idx, dim=0)
    chunk_list_current_q = torch.split(current_q.cpu(), chunk_idx, dim=0)
    chunk_list_current_mu = torch.split(current_mu.cpu(), chunk_idx, dim=0)
    chunk_list_current_sigma = torch.split(current_sigma.cpu(), chunk_idx, dim=0)
    arg_list = [dict(em_iter = EM_Iter) for _ in range(len(chunk_list_org_q))]
    with multiprocessing.Pool(processes=2) as pool:
        result_lst = pool.map(_compute_EM_algorithm, zip(chunk_list_org_q, chunk_list_org_mu, chunk_list_org_sigma, chunk_list_current_q, chunk_list_current_mu, chunk_list_current_sigma, arg_list))
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