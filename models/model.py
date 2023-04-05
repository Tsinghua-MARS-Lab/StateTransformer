from TransformerXL.model import *
import torch.nn as nn

_CHECKPOINT_FOR_DOC = "transfo-xl-wt103"
_CONFIG_FOR_DOC = "TransfoXLConfig"

class TransfoXLModelNuPlan(TransfoXLPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = TransfoXLModel(config)
        # added
        # self.goal_cnn_encoder = CNNEncoder(config, in_channels=1)
        # self.map_cnn_encoder = CNNEncoder(config, in_channels=20)
        # self.agents_cnn_encoder = CNNEncoder(config, in_channels=72)
        #
        # self.goal_cnn_downsample = CNNDownSampling(config, in_channels=config.d_head)
        # self.map_cnn_downsample = CNNDownSampling(config, in_channels=config.d_head)
        # self.agents_cnn_downsample = CNNDownSampling(config, in_channels=config.d_head)

        self.use_nsm = False
        if self.use_nsm:
            n_embed = config.d_embed // 4
        else:
            n_embed = config.d_embed // 2 
        self.cnn_downsample = CNNDownSamplingResNet18(n_embed, in_channels=29)
        
        self.intended_m_embed = nn.Sequential(nn.Embedding(num_embeddings=30, embedding_dim=n_embed), nn.Tanh())
        self.current_m_embed = nn.Sequential(nn.Linear(12, n_embed, bias=False), nn.Tanh())
        self.action_m_embed = nn.Sequential(nn.Linear(4, config.d_embed), nn.Tanh())
        
        self.pos_x_decoder = None
        self.pos_y_decoder = None
        self.traj_decoder = None
        self.predict_pose = True
        self.predict_trajectory = True
        
        try:        
            if self.predict_pose:
                self.pos_x_decoder = DecoderResCat(config.d_inner, config.d_embed, out_features=200)  # from -100 to 100
                self.pos_y_decoder = DecoderResCat(config.d_inner, config.d_embed, out_features=200)  # from -100 to 100
            if self.predict_trajectory:    
                self.traj_decoder = DecoderResCat(config.d_inner, config.d_embed, out_features=4)
        
        except:
            pass
        
        self.intended_m_decoder = DecoderResCat(config.d_inner, config.d_embed, out_features=12)
        self.current_m_decoder = DecoderResCat(config.d_inner, config.d_embed, out_features=12)
        # end of added
        # Initialize weights and apply final processing
        self.post_init()

    def prepare_raster(self, images):
        # raster_images = np.array(images, dtype=np.float32)
        # raster_images = torch.tensor(raster_images, device=device, dtype=torch.float32)
        raster_images = images.permute(0, 3, 1, 2).contiguous().to(torch.float32)
        # print('debug: ', raster_images.shape)
        return raster_images

    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TransfoXLNuPlanNSMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        # input_ids: Optional[torch.LongTensor] = None,
        intended_maneuver_vector: Optional[torch.LongTensor] = None,
        current_maneuver_vector: Optional[torch.LongTensor] = None,
        action_label: Optional[torch.LongTensor] = None,
        trajectory_label: Optional[torch.LongTensor] = None,
        context_actions:Optional[torch.LongTensor] = None,
        intended_maneuver_label: Optional[torch.LongTensor] = None,
        current_maneuver_label: Optional[torch.LongTensor] = None,
        high_res_raster: Optional[torch.LongTensor] = None,
        low_res_raster: Optional[torch.LongTensor] = None,

        mems: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, TransfoXLNuPlanNSMOutput]:


        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if intended_maneuver_vector is not None and current_maneuver_vector is not None:
            intended_maneuver_embed = self.intended_m_embed(intended_maneuver_vector)  # [bsz, hidden_size]
            # current_maneuver_embed = self.current_m_embed(current_maneuver_vector.float())  # [bsz, hidden_size]
            current_maneuver_embed = self.current_m_embed(current_maneuver_vector)  # [bsz, hidden_size]
        else:
            intended_maneuver_embed = None
            current_maneuver_embed = None

        device = high_res_raster.device

        batch_size, h, w, total_channels = high_res_raster.shape

        high_res_seq = self.cat_raster_seq(high_res_raster.permute(0, 3, 2, 1))
        low_res_seq = self.cat_raster_seq(low_res_raster.permute(0, 3, 2, 1))
        batch_size, context_length, c, h, w = high_res_seq.shape
        # embed with the format of (batchsize*history, n_embed) => (batchsize, history, n_embed): both high and low res => (batchsize, history, 2*n_embed) 
        high_res_embed = self.cnn_downsample(high_res_seq.to(torch.float32).reshape(batch_size*context_length, c, h, w))
        low_res_embed = self.cnn_downsample(low_res_seq.to(torch.float32).reshape(batch_size*context_length, c, h, w))
        high_res_embed = high_res_embed.reshape(batch_size, context_length, -1)
        low_res_embed = low_res_embed.reshape(batch_size, context_length, -1)

        if intended_maneuver_embed is not None and current_maneuver_embed is not None:
            state_embeds = torch.cat((intended_maneuver_embed.unsqueeze(1),
                                    current_maneuver_embed.unsqueeze(1),
                                    high_res_embed,
                                    low_res_embed), dim=-1).to(torch.float32)
        else:
            state_embeds = torch.cat((high_res_embed, 
                                       low_res_embed), dim=-1).to(torch.float32)
        
        action_embeds = self.action_m_embed(context_actions)
        # n_embed is 2/4 multiple because different embeddings are concated togaher at the same timestep.
        n_embed = action_embeds.shape[-1]
        input_embeds = torch.zeros(
            (batch_size, context_length * 2, n_embed),
            dtype=torch.float32,
            device=device
        )
        input_embeds[:, ::2, :] = state_embeds
        input_embeds[:, 1::2, :] = action_embeds

        trajectory_label = trajectory_label[:, 1::2, :] # downsample the 20hz trajectory to 10hz
        pred_length = trajectory_label.shape[1]
        input_embeds = torch.cat([input_embeds, torch.zeros(batch_size, pred_length - 2 * context_length, n_embed)], dim=1)
        transformer_outputs = self.transformer(
            None,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        transformer_outputs_hidden_state = transformer_outputs['last_hidden_state']

        # assert (
        #     self.config.pad_token_id is not None or batch_size == 1
        # ), "Cannot handle batch sizes > 1 if no padding token is defined."

        # if self.config.pad_token_id is None:
        #     sequence_lengths = -1
        # else:
        #     if input_ids is not None:
        #         sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
        #     else:
        #         sequence_lengths = -1
        #         logger.warning(
        #             f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
        #             "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
        #         )
        if intended_maneuver_vector is not None and current_maneuver_vector is not None:
            intended_m_logits = self.intended_m_decoder(transformer_outputs_hidden_state[:, 0, :])
            current_m_logits = self.current_m_decoder(transformer_outputs_hidden_state[:, 1, :])
        else:
            intended_m_logits = None
            current_m_logits = None

        if self.pos_x_decoder is not None:
            pos_x_logits = self.pos_x_decoder(transformer_outputs_hidden_state[:, 2, :])
        else:
            pos_x_logits = None
        
        if self.pos_y_decoder is not None:
            pos_y_logits = self.pos_y_decoder(transformer_outputs_hidden_state[:, 3, :])
        else:
            pos_y_logits = None
        
        if self.traj_decoder is not None:
            # expected shape for pred trajectory is (b, pred_length, 4)
            traj_pred = self.traj_decoder(transformer_outputs_hidden_state)
        else:
            traj_pred = None
        # pooled_intended_m_logits = intended_m_logits[range(batch_size), sequence_lengths]
        # pooled_current_m_logits = current_m_logits[range(batch_size), sequence_lengths]
        # pooled_pos_x_logits = pos_x_logits[range(batch_size), sequence_lengths]
        # pooled_pos_y_logits = pos_y_logits[range(batch_size), sequence_lengths]
        # print('test2: ', pooled_intended_m_logits.shape, pooled_current_m_logits.shape, pooled_pos_x_logits.shape, pooled_pos_y_logits.shape)
        # print('test3" '. pooled_intended_m_logits.view(-1, 12).shape, intended_maneuver_label.view(-1).shape)
        # pooled_logits = logits[range(batch_size), sequence_lengths]

        loss = torch.tensor(0, dtype=torch.float32, device=device)
        self.config_problem_type = 'NuPlan_NSM_SingleStep_Planning'
        if intended_maneuver_label is not None:
            loss_fct = CrossEntropyLoss()
            loss += loss_fct(intended_m_logits.view(-1, 12), intended_maneuver_label.view(-1).long())
            # print('test 1193: ', loss, intended_m_logits, intended_maneuver_label, intended_maneuver_label.dtype)    # 2.8
        else:
            pass
            # print('WARNING: intended_maneuver_label is None')
        if current_maneuver_label is not None:
            loss_fct = MSELoss()
            current_c_confifence = torch.softmax(current_m_logits, dim=-1)
            loss += loss_fct(current_c_confifence.squeeze(), current_maneuver_label.squeeze())

            # print('test 1200: ', loss, current_m_logits, current_maneuver_label)    # 3.3
        else:
            pass
            # print('WARNING: current_maneuver_label is None')

        if action_label is not None:
            if self.pos_x_decoder is not None:
                action_label_x = action_label[:, 0] + torch.tensor([100], device=device)  # original action label from -100 to 100
                loss_fct = CrossEntropyLoss()
                loss += loss_fct(pos_x_logits.view(-1, 200), action_label_x.view(-1)) * 0.2
                loss_fct = SmoothL1Loss()
                pos_x = torch.argmax(pos_x_logits, dim=-1)
                loss += loss_fct(pos_x.float(), action_label_x.float()) * 0.1
            if self.pos_y_decoder is not None:
                action_label_y = action_label[:, 1] + torch.tensor([100], device=device)  # original action label from -100 to 100
                loss_fct = CrossEntropyLoss()
                loss += loss_fct(pos_y_logits.view(-1, 200), action_label_y.view(-1)) * 0.2
                loss_fct = SmoothL1Loss()
                pos_y = torch.argmax(pos_y_logits, dim=-1)
                loss += loss_fct(pos_y.float(), action_label_y.float()) * 0.1                                
        else:
            pass
            # print('WARNING: action_label is None')
        
        if trajectory_label is not None:
            if self.traj_decoder is not None:
                loss_fct = MSELoss(reduction="mean")
                loss += loss_fct(traj_pred, trajectory_label)

        pooled_logits = [intended_m_logits, current_m_logits,
                         pos_x_logits,
                         pos_y_logits]
        # pooled_logits = torch.cat((intended_m_logits.unsqueeze(0),
        #                            current_m_logits.unsqueeze(0),
        #                            pos_x_logits.unsqueeze(0),
        #                            pos_y_logits.unsqueeze(0)), dim=0)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # return TransfoXLSequenceClassifierOutputWithPast(
        #     loss=loss,
        #     logits=pooled_logits,
        #     mems=transformer_outputs.mems,
        #     hidden_states=transformer_outputs.hidden_states,
        #     attentions=transformer_outputs.attentions,
        # )

        return TransfoXLNuPlanNSMOutput(
            loss=loss,
            logits=intended_m_logits.cpu() if intended_m_logits is not None else None,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            all_logits=pooled_logits
        )
        
    def cat_raster_seq(self, raster:Optional[torch.LongTensor]):
        """
        input raster can be either high resolution raster or low resolution raster
        expected input size: [bacthsize, channel, h, w], and channel is consisted of goal(1d)+roadtype(20d)+agenttype*time(8*9d)
        """
        framenum = 9 # default for 2s and 5hz sampling
        b, c, h, w = raster.shape
        agent_type = 8
        road_type = 20

        goal_raster = raster[:, 0, :, :].reshape(b, 1, h, w)
        road_ratser = raster[:, 1:21, :, :]
        result = torch.zeros((b, framenum, agent_type + road_type + 1, h, w))
        for i in range(framenum):
            agent_raster = raster[:, 1 + road_type + i::framenum, :, :]
            raster_i = torch.cat([goal_raster, road_ratser, agent_raster], dim = 1) # expected format (b, 1+20+8, h, w)
            result[:, i, :, :, :] = raster_i
        # return format (batchsize, history_frame_number, channels_per_frame, h, w)
        return result

if __name__ == '__main__':
    model = TransfoXLModelNuPlan.from_pretrained('transfo-xl-wt103')
    result = model.forward(
        intended_maneuver_label=None,
        intended_maneuver_vector=None,
        current_maneuver_label=None,
        current_maneuver_vector=None,
        action_label=None,
        trajectory_label=torch.zeros(2, 160, 4),
        context_actions=torch.zeros(2, 9, 4),
        high_res_raster=torch.zeros(2, 224, 224, 93),
        low_res_raster=torch.zeros(2, 224, 224, 93),
        mems=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    )
    print("done")