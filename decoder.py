# Decoder-v1

import math
import torch
import torch.nn as nn

class Decoder(nn.Module):
    r""" This is reverse of Segment.
    Given the learned parameters model.x, model.y apply them to a 
    distribution of x values normalized between 0 - 1 to calculate y
    This should recreate the function plot y = f(x)
    Primary use is to return flatten image values
    
    Args:
        image_shape: [depth, width, height]
        segment_x, segment_y: Segment coordinates

    Shape:
        - Input: 
        image_shape: iD array
        segment_x.shape: [n_batch, n_segments+1]
        segment_y.shape: Same as segment_x.shape

        - Output: [n_batch, depth*width*height]

    Attributes:
        No learnable parameters

    """
    def __init__(self, image_shape):
        super().__init__()
        self.image_size = math.prod(image_shape)
        x_in = torch.arange(1, self.image_size+1)/self.image_size #normalized
        # unsqueeze to add a dimension for n_batch and n_segments
        self.x_in_nd = x_in.unsqueeze(-1).unsqueeze(-1)
        # x_in_nd.shape = [image_size, 1, 1]

    def __calc_mask__(self):
        # x_in_nd.shape = [image_size, 1, 1]
        # create mask to find out which one segment each x belongs to.
        # one x value should only fit in one segment
        lt = torch.lt(self.x_in_nd, self.x[:, 1:]) 
        ge = torch.ge(self.x_in_nd, self.x[:, :-1]) 
        mask = (lt & ge)
        #mask.shape = [image_size, batch_size, n_segments]

        # This mask doesn't include x where x is below 1st segment start
        # or after last segment end
        # we create new mask to capture the x values beyond segments
        #mask_lt.shape = [image_size, batch_size, 1]
        mask_lt = torch.lt(self.x_in_nd,self.x[:, 0:1]) #x less than 1st segment
        # then do OR with mask so these are included for prediction.
        mask[:, :, 0:1] = mask[:, :, 0:1] | mask_lt

        #do the same for last x of segment
        mask_ge = torch.ge(self.x_in_nd, self.x[:, -1:]) 
        mask[:, :, -1:] = mask[:, :, -1:] | mask_ge

        return mask

    def __fix_x_order__(self, segment_x, segment_y):
        # if the next param value is less than previous then replace with previous.
        #to work around grad passing restrictions
        #use Out-of-Place Operation + In-place Copy
        mask_lt = torch.lt(segment_x[:, 1:], segment_x[:, :-1])
        while(torch.any(mask_lt)):
            temp_x = segment_x.clone()
            temp_y = segment_y.clone()
            temp_x[:, 1:][mask_lt] = segment_x[:, :-1].clone()[mask_lt]  # Use .clone() here! 
            temp_y[:, 1:][mask_lt] = segment_y[:, :-1].clone()[mask_lt]  # Use .clone() here! 
            #self.x[:] = temp 
            segment_x = temp_x
            segment_y = temp_y
            mask_lt = torch.lt(segment_x[:, 1:], segment_x[:, :-1])
        return segment_x, segment_y

    def forward(self, segment_x, segment_y):
        #segment_x.shape,segment_y.shape  = [n_batch, n_segments+1]
        #self.batch_size = segment_x.shape[0] #we don't need this variable
        self.x, self.y = self.__fix_x_order__(segment_x, segment_y)
       
        mask = self.__calc_mask__()

        divider = (self.x[:, 1:]-self.x[:, :-1])
        #The ratio can get -inf or inf. we need to protect against it
        divider[divider == 0.] = 0.0001

        # Ratio is the segment (y2-y1)/(x2-x1) ratio 
        ratio = (self.y[:, 1:]-self.y[:, :-1])/divider

        ypred = ratio*mask*(self.x_in_nd - self.x[:,:-1]) + mask * self.y[:,:-1]
        #ypred.shape = [image_size, batch_size, n_segments]
        # As only 1 segment should have value for x we can sum on that dim
        ypred = ypred.sum(2)
        # Reshape ypred so it matches input
        #ypred = ypred.reshape(ypred.shape[1], ypred.shape[0])
        ypred = ypred.permute(1,0)
        return ypred