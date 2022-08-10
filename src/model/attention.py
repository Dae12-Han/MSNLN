
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils
from model import common
from model.utils.tools import extract_image_patches,\
    reduce_mean, reduce_sum, same_padding

#multi-scale non-local attention
class MultiScaleAttention(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, softmax_scale=10, average=True, conv=common.default_conv):
        super(MultiScaleAttention, self).__init__()
        self.conv_match1 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match2 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act = nn.PReLU())
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())
        self.kernel_size = ksize
        
    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input)
        N, C, H, W = x_embed_1.shape

        xe1 = extract_image_patches(x_embed_1, ksizes=[self.kernel_size, self.kernel_size],
                                      strides=[1, 1],
                                      rates=[1, 1],
                                      padding='same')  # [N, C*k*k, L], L = H*W

        xe1 = xe1.contiguous().view(N, C, self.kernel_size, self.kernel_size, -1)  # shape: [N, C, k, k, L]
        xe1 = xe1.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        xe1_groups = torch.split(xe1, 1, dim=0)

        xe2 = extract_image_patches(x_embed_2, ksizes=[self.kernel_size, self.kernel_size],
                                    strides=[1, 1],
                                    rates=[1, 1],
                                    padding='same')  # [N, C*k*k, L]

        xe2 = xe2.contiguous().view(N, C, self.kernel_size, self.kernel_size, -1)  # shape: [N, C, k, k, L]
        xe2 = xe2.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        xe2_groups = torch.split(xe2, 1, dim=0) # N * [L, C, k, k]

        s = []
        for p1, p2 in zip(xe1_groups, xe2_groups):
            # shape of pp, [C, L, k*k]*[C, k*k, L] = [C, L, L]
            pp = torch.matmul(p1[0].contiguous().view(C, -1, self.kernel_size*self.kernel_size),
                              p2[0].contiguous().view(C, self.kernel_size*self.kernel_size, -1))

            #s.append(reduce_sum(pp, 0))   # appending a tensor of size [L, L]
            pp = torch.sum(pp, dim=0, keepdim=True)
            s.append(pp)
        score = torch.cat(s, dim=0)     # [N, L, L], L=H*W

        #x_embed_1 = x_embed_1.permute(0,2,3,1).view((N,H*W,C))
        #x_embed_2 = x_embed_2.view(N,C,H*W)
        #score = torch.matmul(x_embed_1, x_embed_2)  # shape of score = N, H*W, H*W
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.contiguous().view(N,-1,H*W).permute(0,2,1)  # shape of x_assembly = N, H*W, C
        x_final = torch.matmul(score, x_assembly) # shape of x_final = N, H*W, C
        return x_final.permute(0,2,1).contiguous().view(N,-1,H,W)


#in-scale non-local attention
class NonLocalAttention(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True, conv=common.default_conv):
        super(NonLocalAttention, self).__init__()
        self.conv_match1 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match2 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act = nn.PReLU())
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1,bn=False, act=nn.PReLU())
        
    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input)

        N,C,H,W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0,2,3,1).view((N,H*W,C))
        x_embed_2 = x_embed_2.view(N,C,H*W)
        score = torch.matmul(x_embed_1, x_embed_2)
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(N,-1,H*W).permute(0,2,1)
        x_final = torch.matmul(score, x_assembly)
        return x_final.permute(0,2,1).view(N,-1,H,W)


#cross-scale non-local attention
class CrossScaleAttention(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True, conv=common.default_conv):
        super(CrossScaleAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.softmax_scale = softmax_scale
        
        self.scale = scale
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_1 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match_2 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())
        #self.register_buffer('fuse_weight', fuse_weight)

        

    def forward(self, input):
        #get embedding
        embed_w = self.conv_assembly(input)
        match_input = self.conv_match_1(input)
        
        # b*c*h*w
        shape_input = list(embed_w.size())   # b*c*h*w
        input_groups = torch.split(match_input,1,dim=0)
        # kernel size on input for matching 
        kernel = self.scale*self.ksize
        
        # raw_w is extracted for reconstruction 
        raw_w = extract_image_patches(embed_w, ksizes=[kernel, kernel],
                                      strides=[self.stride*self.scale,self.stride*self.scale],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.contiguous().view(shape_input[0], shape_input[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)
        
    
        # downscaling X to form Y for cross-scale matching
        ref  = F.interpolate(input, scale_factor=1./self.scale, mode='bilinear')
        ref = self.conv_match_2(ref)
        w = extract_image_patches(ref, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        shape_ref = ref.shape
        # w shape: [N, C, k, k, L]
        w = w.contiguous().view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)


        y = []
        scale = self.softmax_scale  
          # 1*1*k*k
        #fuse_weight = self.fuse_weight

        for xi, wi, raw_wi in zip(input_groups, w_groups, raw_w_groups):
            # normalize
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi/ max_wi

            # Compute correlation map
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W] L = shape_ref[2]*shape_ref[3]

            yi = yi.contiguous().view(1,shape_ref[2] * shape_ref[3], shape_input[2], shape_input[3])  # (B=1, C=32*32, H=32, W=32)
            # rescale matching score
            yi = F.softmax(yi*scale, dim=1)
            if self.average == False:
                yi = (yi == yi.max(dim=1,keepdim=True)[0]).float()
            
            # deconv for reconsturction
            wi_center = raw_wi[0]           
            yi = F.conv_transpose2d(yi, wi_center, stride=self.stride*self.scale, padding=self.scale)
            
            yi =yi/6.
            y.append(yi)
      
        y = torch.cat(y, dim=0)
        return y

