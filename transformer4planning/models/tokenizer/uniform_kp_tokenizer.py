import torch


class UniformKPTokenizer:
    """
    Tokenizer for uniformly sample key point proposal from minimal to maximal xy coordinates.
    """
    def __init__(self, num_key_points: list, x_min: float, x_max: float, y_min: float, y_max: float):
        self.num_key_points = num_key_points
        self.key_point_num_on_x, self.key_point_num_on_y = num_key_points
        # uniformly sample key points from minimal to maximal xy coordinates
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def encode(self, key_points, dtype, device=None):
        """
        Encode key points to token ids.
        Return a two-dimensional array with shape (num_key_points, 2).
        """
        x_key_points = torch.linspace(self.x_min, self.x_max, self.key_point_num_on_x, dtype=dtype).to(device)
        y_key_points = torch.linspace(self.y_min, self.y_max, self.key_point_num_on_y, dtype=dtype).to(device)
        x_ids = torch.bucketize(key_points[:, 0].contiguous(), x_key_points) - 1  # [bsz, 1]
        y_ids = torch.bucketize(key_points[:, 1].contiguous(), y_key_points) - 1  # [bsz, 1]
        # Check if any x_ids are out of bounds
        if torch.any(x_ids < 0) or torch.any(x_ids >= self.key_point_num_on_x):
            # print('x id overflow: ', x_ids, self.key_point_num_on_x, self.x_min, self.x_max, key_points[:, 0])
            x_ids = torch.clip(x_ids, 0, self.key_point_num_on_x)

        # Check if any y_ids are out of bounds
        if torch.any(y_ids < 0) or torch.any(y_ids >= self.key_point_num_on_y):
            # print('y id overflow: ', y_ids, self.key_point_num_on_y, self.y_min, self.y_max, key_points[:, 1])
            y_ids = torch.clip(y_ids, 0, self.key_point_num_on_y)

        key_point_ids = x_ids + y_ids * self.key_point_num_on_x
        return key_point_ids

    def decode(self, kp_ids, dtype, device=None):
        """
        Decode token ids back to key points.
        Return a two-dimensional array with shape (num_input_points, 2).
        """
        batch_size = kp_ids.shape[0]  # kp_ids: [bsz]
        key_points = torch.zeros([batch_size, 2], dtype=dtype, device=device)
        x_key_points = torch.linspace(self.x_min, self.x_max, self.key_point_num_on_x, dtype=dtype).to(device)
        y_key_points = torch.linspace(self.y_min, self.y_max, self.key_point_num_on_y, dtype=dtype).to(device)
        x_ids = kp_ids % self.key_point_num_on_y
        y_ids = kp_ids / self.key_point_num_on_y
        key_points[:, 0] = x_key_points[x_ids.long()]
        key_points[:, 1] = y_key_points[y_ids.long()]
        return key_points
