import torch
import torch.nn as nn
import pickle
from matplotlib import path
import numpy as np
eps=1e-9

class Loss(torch.nn.Module):

    def __init__(self, 
                 K=6,
                 future_num_frames=80,
                 loss_function_name='cal_Loss',
                 cls_magin=0.2,
                 cls_th=2.0 ,
                 cls_ignore=0.2,
                 MultiTask=False,
                 auxiliary_params=None):
        super(Loss,self).__init__()
        self.future_frames_num = int(future_num_frames)
        self.K = K
        self.loss_function = getattr(self, loss_function_name)
        self.cls_magin = cls_magin
        self.cls_th = cls_th
        self.cls_ignore = cls_ignore
        self.margin = 0.2
        self.gamma = 2
        self.huber_theta = 1
        self.prior_weight = 0.5
        self.eps = 1e-8
        self.inf = torch.tensor(1e8).cuda() 
        self.smooth_L1_loss = torch.nn.SmoothL1Loss(reduction= 'none')
        # self.sigmoid = torch.nn.Sigmoid()
        self.alpha = 0.01
        # self.alpha = 1
        self.softmax = torch.nn.Softmax(dim=-1)
        self.KL = torch.nn.KLDivLoss(reduction='none')

        self.epoch = 0
        self.anneal_interval = [10, 30, 45]
        # self.top_K = 4
        self.top_K = 1

        if MultiTask and auxiliary_params is not None:
            self.auxiliary_params = auxiliary_params
        self.weight = torch.ones(self.K).cuda()
        
        self.MultiTask = MultiTask
        self.loss_params = dict()


    def forward(self, pred, data:dict):
        '''
        pred [batch*(target_car + nbrs_car_num), K, future_len, xy_dim(2)]
        target_pred  
        '''
        # if 'aux_outputs' in pred:
        #     mask_aux = epoch//40
        #     if mask_aux > 3:
        #         mask_aux = 3
        #     pred['aux_outputs'] = pred['aux_outputs'][mask_aux:]
        # if epoch > 40 and 'precls' in pred:
        #     pred.pop('precls')
        return self.loss_wrapper(pred, data)

    def step(self):
        self.epoch += 1
        for i, supremum in enumerate(self.anneal_interval):
            if self.epoch < supremum:
                # self.top_K = 4-i
                # break 
                self.top_K = 1
                break

    def margin_loss(self, score, dis_mat, index, endpoint_exist_mask):
        '''
            params:
                score[batch, num agents, K]: confidence of each traj
                dis_mat[batch, num agents, K]: the score beased on endpoint 
                index[batch, num agents]: index of traj choosed as the best traj
                endpoint_exist_mask[batch, num agents,1]: contain the info of which traj has endpoint
        '''
        #FIXME need to fix
        index = index.unsqueeze(-1)#[batch, num_agent, 1]
        best_dis = torch.gather(dis_mat, dim=-1, index=index)
        best_score = torch.gather(score, dim=-1, index=index)
        # filter 2m prediction, to near traj, best one
        valid_sample_mask = (best_dis <= self.cls_th) * (dis_mat- best_dis >= self.cls_ignore) * (dis_mat != best_dis) #[batch, num_agent, K]
        zero_mat = torch.zeros_like(score)
        cls_loss = torch.max(zero_mat,(score+self.margin-best_score))#[batch, num agents, K]
        cls_loss = cls_loss*valid_sample_mask*endpoint_exist_mask #some sample can't be choosed as the sample and the traj with no gt
        # calculate the mean
        valid_sample_num = valid_sample_mask.sum(dim=-1) #[batch, num agents]
        valid_sample_num.masked_fill_(valid_sample_num==0,1.0)
        cls_loss = cls_loss.sum(dim=-1)/valid_sample_num #[batch, num agents]
        # cls_loss = cls_loss.sum(dim=-1)/(self.K-1)#[batch, num agents]
        cls_loss = cls_loss.sum(dim=-1)/endpoint_exist_mask.squeeze(dim=-1).sum(-1) #[batch]
        return cls_loss.mean()

    def maxEntropyLoss(self, score, dis_mat, endpoint_exist_mask):
        '''
            params:
                score[batch, num agents, K]: confidence of each traj
                dis_mat[batch, num agents, K]: the score beased on endpoint 
                endpoint_exist_mask[batch, num agents,1]: contain the info of which traj has endpoint
        '''
        score = score.log()
        # score = self.softmax(score).log()
        dis_mat = dis_mat.detach()
        target = self.softmax(-dis_mat/self.alpha)
        cls_loss = self.KL(score, target)
        # cls_loss = self._cross_entropy(score, target)
        # return cls_loss
        cls_loss = cls_loss*endpoint_exist_mask
        # calculate the mean
        cls_loss = cls_loss.sum(dim=-1) #[batch, number of agents]
        cls_loss = cls_loss.sum(dim=-1)/endpoint_exist_mask.squeeze(dim=-1).sum(-1) #[batch]
        return cls_loss.mean()

    def _cross_entropy(self, score, target, gamma=2):
        ce_loss = -score*target*(self.weight**gamma)
        return ce_loss

    def cross_entropy_discrete(self, score, target, endpoint_exist_mask):
        '''
            params:
                score[batch, num agents, K]: confidence of each traj
                target[batch, ]: gt_index 
        '''
        endpoint_exist_mask = endpoint_exist_mask.squeeze(dim=-1)
        score = score.permute(0,2,1)
        target = target.unsqueeze(-1).repeat(1, score.shape[-1])
        cls_loss = nn.functional.cross_entropy(score, target, reduction='none')
        cls_loss = cls_loss*endpoint_exist_mask
        # calculate the mean
        cls_loss = cls_loss.sum(dim=-1)/endpoint_exist_mask.sum(-1) #[batch]
        return cls_loss.mean() 

    def huber_loss(self, score, target, expected_traj_index, mask):
        '''
            params:
                score: the predicted traj
                target: the ground truth of traj
                expected_traj_index: decide which traj can be choose from total K trajs.
                mask[batch, numagents, future num frame]: the frame avaliable mask
        '''
        expected_traj_index = expected_traj_index.view(*expected_traj_index.shape,1,1,1).repeat((1,1,1,*score.shape[3:]))#[batch_size, nbrs_num+1, K, 30, 2]
        best_pred = torch.gather(score, dim=-3, index=expected_traj_index)
        # TODO smooth L1 maybe better than hand made huber loss
        reg_loss = self.smooth_L1_loss(best_pred, target).squeeze(dim=-3).mean(-1)
        # mask = mask*(mask[...,-1].unsqueeze(-1).repeat(1,1,self.future_frames_num)) #XXX need to decide whether to remove here!
        reg_loss = (reg_loss*mask).sum(dim=-1) # [batch_size, nbrs_num+1]
        # Start to calculate mean~
        loss_available_frame_num = mask.sum(dim=-1)# [batch_size, nbrs_num+1]
        loss_available_frame_num_nan = loss_available_frame_num.masked_fill(loss_available_frame_num==0, 1.0) #prevent nan
        reg_loss = reg_loss/loss_available_frame_num_nan # [batch_size, nbrs_num+1]
        loss_available_traj_mask = (loss_available_frame_num!=0)
        loss_available_traj_num = loss_available_traj_mask.sum(dim=-1)
        # reg_loss = self.gt_prior_weighted_sum(reg_loss, loss_available_traj_num, loss_available_traj_mask).sum(-1)
        reg_loss = reg_loss.sum(dim=-1)/loss_available_traj_num # [batch_size] If reg loss be weighed then don't have to divide n
        
        return reg_loss.mean()

    def divide_loss(self, y, pred, K, mask, conf=None, par_conf=None, verbose=True, inner_product=None):
        '''
            pred: [batch, nbrs_num+1, K, future_num_frames, 2]
            y:    [batch, nbrs_num+1, 1, future_num_frames, 2] 
            conf: [batch, nbrs_num+1, K]

            return loss and missrate
        '''
        losses, MRs = [], []
        assert K >= self.num_region &  ~(K%self.num_region) ,\
                'divide only support the condition where K>{} and K:{}can be divided by num region:{}'.format(self.num_region, self.K, self.num_region)
        # select the nearest prediction
        concentrate_num = K//self.num_region
        region_gt, proposal_mask = self.get_gt_region(y, concentrate_num)         
        y_endpoint, end_point_index, endpoint_exist_mask = self.get_last_nonzero_index_for_each_trajectory(y)
        pred_endpoint =  torch.gather(pred.detach() , dim=-2, index=end_point_index.repeat(1,1,self.K,1,1)).squeeze(-2)  
        dis_mat = torch.norm((y_endpoint - pred_endpoint), p=2, dim=-1) #[batch, nbrs_num+1, K] #TODO fix this 
        dis_mat = dis_mat.masked_fill_(proposal_mask.unsqueeze(dim=1).repeat(1,dis_mat.shape[1],1)==0, self.inf)
        index = torch.argmin(dis_mat, dim=-1) #[batch, nbrs_num+1, 1]

        # assert dis_mat.min() < self.inf, 'no selected index!!!! check!!!'
        if not (dis_mat.min() < self.inf):
            import pdb; pdb.set_trace()
        reg_loss = self.huber_loss(pred, y, index, mask) # Huber Loss
        miss_rate = self.cal_ego_miss_rate(y_endpoint, pred_endpoint, endpoint_exist_mask) if verbose else 0
        cls_miss_rate = self.cal_ego_cls_miss_rate(region_gt, conf) if verbose else 0
        miss_rate = {'MR':miss_rate,'CMR':cls_miss_rate}
        
        if conf.shape[-1] == self.num_region:
            cls_loss = self.cross_entropy_discrete(conf, region_gt, endpoint_exist_mask) 
        elif par_conf is not None :
            par_cls_loss = self.cross_entropy_discrete(par_conf, region_gt, endpoint_exist_mask)         
            cls_loss = self.maxEntropyLoss(conf, dis_mat, endpoint_exist_mask)
        else:
            cls_loss = self.maxEntropyLoss(conf, dis_mat, endpoint_exist_mask) 
        
        losses = {'reg_loss':{'loss':reg_loss, 'kpi': miss_rate['MR']}, 'cls_loss':{'loss':cls_loss, 'kpi': miss_rate['CMR']}}
        if par_conf is not None:
            losses.update({'Pcls_loss':{'loss':par_cls_loss, 'kpi': miss_rate['CMR']}})

        
        return  losses, miss_rate
    
    def cal_Loss(self, y, pred, K, mask, par_conf=None, conf=None, verbose=True, **kargs):
        '''
            pred: [batch, nbrs_num+1, K, future_num_frames, 2]
            y:    [batch, nbrs_num+1, 1, future_num_frames, 2] 
            conf: [batch, nbrs_num+1, K]
        '''
        # select the nearest prediction
        y_endpoint, end_point_index, endpoint_exist_mask = self.get_last_nonzero_index_for_each_trajectory(y)
        pred_endpoint =  torch.gather(pred, dim=-2, index=end_point_index.repeat(1,1,self.K,1,1)).squeeze(-2)   
        dis_mat = torch.norm((y_endpoint - pred_endpoint), p=2, dim=-1) #[batch, nbrs_num+1, K] #TODO fix this 
        index = torch.argmin(dis_mat, dim=-1) #[batch, nbrs_num+1, 1]
        # cls_loss = self.margin_loss(conf, dis_mat, index, endpoint_exist_mask) if conf is not None else None # Margin Loss
        cls_loss = self.maxEntropyLoss(conf, dis_mat, endpoint_exist_mask)# MaxEntropyLoss
        reg_loss = self.huber_loss(pred, y, index, mask) # Huber Loss
        # miss_rate = self.cal_miss_rate(y_endpoint, pred_endpoint, endpoint_exist_mask)
        miss_rate = self.cal_ego_miss_rate(y_endpoint, pred_endpoint, endpoint_exist_mask) if verbose else 0
        miss_rate = {'MR':miss_rate,'CMR':None}
        # losses = {'reg_loss':{'loss':reg_loss, 'kpi': miss_rate['MR']}}
        losses = {'reg_loss':{'loss':reg_loss, 'kpi': miss_rate['MR']}, 'cls_loss':{'loss':cls_loss, 'kpi': None}}
        return losses, miss_rate

    def prepare_groundtruth(self, data, ego_only=False):

        nbrs_gt = data['nbrs_groundtruth']
        self.batch_size = data['groundtruth'].shape[0] 
        target_y = data['groundtruth'].view(self.batch_size, 1, 1, self.future_frames_num, 2).cumsum(dim=-2) 
        target_y_mask = data['groundtruth_availabilities']
        nbrs_y = nbrs_gt[:,:,:,:2].view(self.batch_size, -1, 1, self.future_frames_num, 2).cumsum(dim=-2)
        nbrs_y_mask = nbrs_gt[:,:,:,2].view(self.batch_size,-1,self.future_frames_num)   
        if ego_only:     
            y = target_y
            y_mask = target_y_mask
        else:
            y = torch.cat([target_y,nbrs_y], dim=1)
            y_mask = torch.cat([target_y_mask, nbrs_y_mask], dim=1)
        return y, y_mask       

    def loss_wrapper(self, model_outputs, future_traj: torch.Tensor):
        '''
            future_traj: torch.tensor bs, nfuture, 2
        '''
        # print(data.keys())

        pred, conf = model_outputs
        # pred, conf = model_outputs['pred_coords'], model_outputs['pred_logits']

        # ego_only= False if pred.shape[1]>1 else True
        # y, y_mask = self.prepare_groundtruth(data, ego_only)
        bs = future_traj.shape[0] 
        y = future_traj.view(bs, -1, 1, self.future_frames_num, 2)
        y_mask = ((y!=-1).sum(-1)>0).view(bs, -1, self.future_frames_num)
        
        # par_conf = model_outputs['precls'] if 'precls' in model_outputs else None
        # inner_product = model_outputs['inner_product'] if 'inner_product' in model_outputs else None        
        losses, miss_rate = self.loss_function(y, pred, self.K, y_mask, conf=conf, par_conf=None, inner_product=None) #target and neighbours loss

        total_loss, loss_text  = self.multi_task_gather(losses)
        return total_loss, loss_text, miss_rate

    def get_last_nonzero_index_for_each_trajectory(self,traj_data):
        '''
        ground_truth:[batchsize, agents number, 1/6, future num frame, 2]
        '''
        traj_data_index = (traj_data!=0).sum(dim=-1) #[batchsize, agents number, K, future number frames]
        non_zero_index = (traj_data_index!=0).sum(dim=-1) - 1 #index start from 0 [batchsize, agents number, K]
        endpoint_exist_mask = (non_zero_index>=0) #the endpoint exist mask
        non_zero_index = non_zero_index*endpoint_exist_mask
        non_zero_index = non_zero_index.view(*non_zero_index.shape,1,1).repeat((1,1,1,1,2))
        endpoint = torch.gather(traj_data, dim=-2, index=non_zero_index).squeeze(-2)
        return (endpoint, 
                non_zero_index, 
                endpoint_exist_mask#[batch, nun_agent,1]
            )

    def gt_prior_weighted_sum(self, input, endpoint_exist_num, traj_mask):
        nbrs_weight = (1-self.prior_weight)/(endpoint_exist_num-1+self.eps).float() 
        weight = torch.empty_like(input)
        weight[:,0] = self.prior_weight
        weight[:,1:] = nbrs_weight.unsqueeze(-1)
        weight = weight*traj_mask
        input = input*weight
        return input
    
    def cal_miss_rate(self, target_endpoint, prediction_endpoint, endpoint_exist_mask):
        '''
            prediction_endpoint: [batch, nbrs_num+1, K, 2]
            target_endpoint:    [batch, nbrs_num+1, 1, 2] 
        '''

        inside_mask = (torch.norm(target_endpoint-prediction_endpoint,p=2,dim=-1) < 2)
        inside_mask = inside_mask*endpoint_exist_mask
        inside_mask = (inside_mask.sum(dim=(-1)) !=0)
        miss_rate   = 1 - inside_mask.sum()/endpoint_exist_mask.sum().float()
        return miss_rate
    
    def get_gt_region(self, y, concentrate_num=6):
        gt = y[:,0,...,-1,:].reshape(-1,2).cpu()
        '''
            gt:[batch_size/batch_size*(num_agents),2]
        '''

        region_index = torch.zeros(gt.shape[0]).cuda().long()
        proposal_mask = torch.empty((gt.shape[0], self.K), dtype=torch.bool).cuda()
        proposal_mask[...] = False
        for i, region in enumerate(self.regions):
            contain_mask = region.contains_points(gt)
            contain_idex = np.nonzero(contain_mask)
            region_index[contain_idex] = i 
            active_slice = slice((i*concentrate_num),((i+1)*concentrate_num))
            proposal_mask[contain_idex, active_slice] = True

        return region_index, proposal_mask
    
    def schedule_training(self, losses):
        _cfg = self.loss_params
        if 'cls_loss_switch' in _cfg:
            losses['cls_loss']['loss'] = losses['cls_loss']['loss']*_cfg['cls_loss_switch']
        if 'reg_loss_switch' in _cfg:
            losses['reg_loss']['loss'] = losses['reg_loss']['loss']*_cfg['reg_loss_switch']
        if 'par_cls_loss_switch' in _cfg and "par_cls_loss" in losses :
            losses['par_cls_loss']['loss'] = losses['par_cls_loss']['loss']*_cfg['par_cls_loss_switch']
        return losses
    
    def multi_task_gather(self, losses):

        losses = self.schedule_training(losses)
        total_loss = 0
        # Total Loss
        if self.MultiTask:
            for i, loss in enumerate(losses.values()):
                loss = loss['loss']
                # ? which kinds of multi task loss suits our task.s
                total_loss+= (2*self.auxiliary_params[i])**-2*loss + torch.log(self.auxiliary_params[i]**2+1)
        elif 'KPI' in self.loss_params and self.loss_params['KPI']:
            for i, loss in enumerate(losses.values()):
                loss, kpi = loss['loss'], loss['kpi']
                # ! since the kpi is to measure how bad the model. unlike noraml is how good of the model.
                total_loss+= -(kpi)**self.gamma*torch.log(1-kpi+self.eps)*loss 
        elif len(losses) < 2:
            for i, loss in enumerate(losses.values()):
                total_loss = loss['loss']
        else:
            total_loss = 5*losses['reg_loss']['loss'] + 0.1*losses['cls_loss']['loss']

        losses_text = ''
        for loss_name in losses:
            loss_text = loss_name.split('_')[0]
            losses_text += loss_text+':{:.3f}-'.format(losses[loss_name]['loss'].item())            
        
        return total_loss, losses_text

    def intermidate_loss_gather(self, model_outputs, losses, y, y_mask):
        
        for loss_name in losses:
            losses[loss_name]['loss'] = [losses[loss_name]['loss'],]
        
        for aux_outputs in model_outputs['aux_outputs']:
            aux_pred, aux_conf = aux_outputs['pred_coords'], aux_outputs['pred_logits']
            aux_parconf = aux_outputs['precls'] if 'precls' in aux_outputs else None
            aux_inner_product = aux_outputs['inner_product'] if 'inner_product' in aux_outputs else None           
            aux_losses, _ = self.loss_function(y, aux_pred, self.K, y_mask, conf=aux_conf, par_conf=aux_parconf,verbose=False, inner_product=aux_inner_product) 
            
            for loss_name in losses:
                losses[loss_name]['loss'].append(aux_losses[loss_name]['loss'])

        for loss_name in losses:
            losses[loss_name]['loss'] = sum(losses[loss_name]['loss'])/ len(losses[loss_name]['loss'])
        
        
        return losses
        
    def cal_ego_cls_miss_rate(self, region_gt, conf):
        '''
            region_gt: [batch]
            conf:    [batch, nbrs_num+1 or 1, K] 
        '''
        ego_conf = conf[:,0,:]
        ego_conf = ego_conf.reshape(self.batch_size, self.num_region,-1).sum(-1)
        ego_region = torch.argmax(ego_conf,dim=-1)
        num_right = (ego_region == region_gt).sum()
        miss_rate   = (1 - float(num_right)/float(region_gt.shape[0]))
        return miss_rate
    
    def cal_ego_miss_rate(self, target_endpoint, prediction_endpoint, endpoint_exist_mask):
        '''
            prediction_endpoint: [batch, nbrs_num+1, K, 2]
            target_endpoint:    [batch, nbrs_num+1, 1, 2] 
        '''

        inside_mask = (torch.norm(target_endpoint-prediction_endpoint,p=2,dim=-1) < 2.0)
        inside_mask = inside_mask*endpoint_exist_mask
        inside_mask = (inside_mask[:,0].sum(dim=(-1))!=0)
        miss_rate   = 1 - inside_mask.sum()/endpoint_exist_mask[:,0].sum().float()
        return miss_rate

