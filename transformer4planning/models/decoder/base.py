from torch import nn

class TrajectoryDecoder(nn.Module):
    def __init__(self, model_args):
        """
        config include the model dimension infomation and model_args.
        model_args include other config items.
        """
        super().__init__()
        self.loss_fn = model_args.loss_fn
        self.k = int(model_args.k) # k means the number of future trajectories to predict
        self.out_features = 4 if model_args.predict_yaw else 2
        self.ar_future_interval = model_args.ar_future_interval
        self.model_args = model_args
        if 'mse' in self.model_args.loss_fn:
            self.loss_fct = nn.MSELoss(reduction="mean")
        elif 'l1' in self.model_args.loss_fn:
            self.loss_fct = nn.SmoothL1Loss()
        
    def forward(self, x):
        raise NotImplementedError
    
    def compute_loss(self, pred, label):
        raise NotImplementedError