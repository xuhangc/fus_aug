import functools
import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

from torch.optim.optimizer import Optimizer

class Adam16(Optimizer):
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0):
    defaults = dict(lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay)
    params = list(params)
    super(Adam16, self).__init__(params, defaults)

  # Safety modification to make sure we floatify our state
  def load_state_dict(self, state_dict):
    super(Adam16, self).load_state_dict(state_dict)
    for group in self.param_groups:
      for p in group['params']:
        self.state[p]['exp_avg'] = self.state[p]['exp_avg'].float()
        self.state[p]['exp_avg_sq'] = self.state[p]['exp_avg_sq'].float()
        self.state[p]['fp32_p'] = self.state[p]['fp32_p'].float()

  def step(self, closure=None):
    """Performs a single optimization step.
    Arguments:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        grad = p.grad.data.float()
        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          # Exponential moving average of gradient values
          state['exp_avg'] = grad.new().resize_as_(grad).zero_()
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
          # Fp32 copy of the weights
          state['fp32_p'] = p.data.float()

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
          grad = grad.add(group['weight_decay'], state['fp32_p'])

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

        denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

        state['fp32_p'].addcdiv_(-step_size, exp_avg, denom)
        p.data = state['fp32_p'].half()

    return loss
  

def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())

# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x

# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs

# Convenience passthrough function
class identity(nn.Module):
  def forward(self, input):
    return input

# Spectral normalization base class
class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))

  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values;
  # note that these buffers are just for logging and are not used in training.
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv
    return self.weight / svs[0]

# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True,
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
  def forward_wo_sn(self, x):
    return F.conv2d(x, self.weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups)


# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)


# Embedding layer with spectral norm
# We use num_embeddings as the dim instead of embedding_dim here
# for convenience sake
class SNEmbedding(nn.Embedding, SN):
  def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
               max_norm=None, norm_type=2, scale_grad_by_freq=False,
               sparse=False, _weight=None,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                          max_norm, norm_type, scale_grad_by_freq,
                          sparse, _weight)


    SN.__init__(self, num_svs, num_itrs, num_embeddings, eps=eps)

  def forward(self, x):
    return F.embedding(x, self.W_())


  #def forward(self, x):
  #  return  F.embedding(x, self.weight)

# A non-local block as used in SA-GAN
# Note that the implementation as described in the paper is largely incorrect;
# refer to the released code for the actual implementation.
class Attention(nn.Module):
  def __init__(self, ch, which_conv=SNConv2d, name='attention'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.which_conv = which_conv
    self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool2d(self.phi(x), [2,2])
    g = F.max_pool2d(self.g(x), [2,2])
    # Perform reshapes
    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
    phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
    g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
    return self.gamma * o + x


# Fused batchnorm op
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
  # Apply scale and shift--if gain and bias are provided, fuse them here
  # Prepare scale
  scale = torch.rsqrt(var + eps)
  # If a gain is provided, use it
  if gain is not None:
    scale = scale * gain
  # Prepare shift
  shift = mean * scale
  # If bias is provided, use it
  if bias is not None:
    shift = shift - bias
  return x * scale - shift
  #return ((x - mean) / ((var + eps) ** 0.5)) * gain + bias # The unfused way.


# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
  # Cast x to float32 if necessary
  float_x = x.float()
  # Calculate expected value of x (m) and expected value of x**2 (m2)
  # Mean of x
  m = torch.mean(float_x, [0, 2, 3], keepdim=True)
  # Mean of x squared
  m2 = torch.mean(float_x ** 2, [0, 2, 3], keepdim=True)
  # Calculate variance as mean of squared minus mean squared.
  var = (m2 - m **2)
  # Cast back to float 16 if necessary
  var = var.type(x.type())
  m = m.type(x.type())
  # Return mean and variance for updating stored mean/var if requested
  if return_mean_var:
    return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
  else:
    return fused_bn(x, m, var, gain, bias, eps)


# My batchnorm, supports standing stats
class myBN(nn.Module):
  def __init__(self, num_channels, eps=1e-5, momentum=0.1):
    super(myBN, self).__init__()
    # momentum for updating running stats
    self.momentum = momentum
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Register buffers
    self.register_buffer('stored_mean', torch.zeros(num_channels))
    self.register_buffer('stored_var',  torch.ones(num_channels))
    self.register_buffer('accumulation_counter', torch.zeros(1))
    # Accumulate running means and vars
    self.accumulate_standing = False

  # reset standing stats
  def reset_stats(self):
    self.stored_mean[:] = 0
    self.stored_var[:] = 0
    self.accumulation_counter[:] = 0

  def forward(self, x, gain, bias):
    if self.training:
      out, mean, var = manual_bn(x, gain, bias, return_mean_var=True, eps=self.eps)
      # If accumulating standing stats, increment them
      if self.accumulate_standing:
        self.stored_mean[:] = self.stored_mean + mean.data
        self.stored_var[:] = self.stored_var + var.data
        self.accumulation_counter += 1.0
      # If not accumulating standing stats, take running averages
      else:
        self.stored_mean[:] = self.stored_mean * (1 - self.momentum) + mean * self.momentum
        self.stored_var[:] = self.stored_var * (1 - self.momentum) + var * self.momentum
      return out
    # If not in training mode, use the stored statistics
    else:
      mean = self.stored_mean.view(1, -1, 1, 1)
      var = self.stored_var.view(1, -1, 1, 1)
      # If using standing stats, divide them by the accumulation counter
      if self.accumulate_standing:
        mean = mean / self.accumulation_counter
        var = var / self.accumulation_counter
      return fused_bn(x, mean, var, gain, bias, self.eps)


# Simple function to handle groupnorm norm stylization
def groupnorm(x, norm_style):
  # If number of channels specified in norm_style:
  if 'ch' in norm_style:
    ch = int(norm_style.split('_')[-1])
    groups = max(int(x.shape[1]) // ch, 1)
  # If number of groups specified in norm style
  elif 'grp' in norm_style:
    groups = int(norm_style.split('_')[-1])
  # If neither, default to groups = 16
  else:
    groups = 16
  return F.group_norm(x, groups)

# Class-conditional bn
# output size is the number of channels, input size is for the linear layers
# Andy's Note: this class feels messy but I'm not really sure how to clean it up
# Suggestions welcome! (By which I mean, refactor this and make a pull request
# if you want to make this more readable/usable).
class ccbn(nn.Module):
  def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1,
               cross_replica=False, mybn=False, norm_style='bn', self_modulation=False):
    super(ccbn, self).__init__()
    self.output_size, self.input_size = output_size, input_size
    # Prepare gain and bias layers
    if self_modulation:
        self.gain = nn.Sequential(which_linear(input_size,input_size, bias=True ), nn.ReLU(), which_linear(input_size, output_size, bias=False))
        self.bias = nn.Sequential(which_linear(input_size,input_size, bias=True ), nn.ReLU(), which_linear(input_size, output_size, bias=False))
    else:
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Use cross-replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    # Norm style?
    self.norm_style = norm_style

    #if self.cross_replica:
    #  self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
    if self.mybn:
      self.bn = myBN(output_size, self.eps, self.momentum)
    elif self.norm_style in ['bn', 'in']:
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size))


  def forward(self, x, y):
    # Calculate class-conditional gains and biases
    gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
    bias = self.bias(y).view(y.size(0), -1, 1, 1)
    # If using my batchnorm
    if self.mybn or self.cross_replica:
      return self.bn(x, gain=gain, bias=bias)
    # else:
    else:
      if self.norm_style == 'bn':
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'in':
        out = F.instance_norm(x, self.stored_mean, self.stored_var, None, None,
                          self.training, 0.1, self.eps)
      elif self.norm_style == 'gn':
        out = groupnorm(x, self.normstyle)
      elif self.norm_style == 'nonorm':
        out = x
      return out * gain + bias
  def extra_repr(self):
    s = 'out: {output_size}, in: {input_size},'
    s +=' cross_replica={cross_replica}'
    return s.format(**self.__dict__)


# Normal, non-class-conditional BN
class bn(nn.Module):
  def __init__(self, output_size,  eps=1e-5, momentum=0.1,
                cross_replica=False, mybn=False):
    super(bn, self).__init__()
    self.output_size= output_size
    # Prepare gain and bias layers
    self.gain = P(torch.ones(output_size), requires_grad=True)
    self.bias = P(torch.zeros(output_size), requires_grad=True)
    # epsilon to avoid dividing by 0
    self.eps = eps
    # Momentum
    self.momentum = momentum
    # Use cross-replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn

    if self.cross_replica:
      self.bn = SyncBN2d(output_size, eps=self.eps, momentum=self.momentum, affine=False)
    elif mybn:
      self.bn = myBN(output_size, self.eps, self.momentum)
     # Register buffers if neither of the above
    else:
      self.register_buffer('stored_mean', torch.zeros(output_size))
      self.register_buffer('stored_var',  torch.ones(output_size))

  def forward(self, x, y=None):
    if self.cross_replica or self.mybn:
      gain = self.gain.view(1,-1,1,1)
      bias = self.bias.view(1,-1,1,1)
      return self.bn(x, gain=gain, bias=bias)
    else:
      return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                          self.bias, self.training, self.momentum, self.eps)


# Generator blocks
# Note that this class assumes the kernel size and padding (and any other
# settings) have been selected in the main generator module and passed in
# through the which_conv arg. Similar rules apply with which_bn (the input
# size [which is actually the number of channels of the conditional info] must
# be preselected)
class GBlock(nn.Module):
  def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, which_bn=bn, activation=None,
               upsample=None):
    super(GBlock, self).__init__()

    self.in_channels, self.out_channels = in_channels, out_channels
    self.which_conv, self.which_bn = which_conv, which_bn
    self.activation = activation
    self.upsample = upsample
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.out_channels)
    self.conv2 = self.which_conv(self.out_channels, self.out_channels)
    self.learnable_sc = in_channels != out_channels or upsample
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels,
                                     kernel_size=1, padding=0)
    # Batchnorm layers
    self.bn1 = self.which_bn(in_channels)
    self.bn2 = self.which_bn(out_channels)
    # upsample layers
    self.upsample = upsample

  def forward(self, x, y):
    h = self.activation(self.bn1(x, y))
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    h = self.conv1(h)
    h = self.activation(self.bn2(h, y))
    h = self.conv2(h)
    if self.learnable_sc:
      x = self.conv_sc(x)
    return h + x

class GBlock2(nn.Module):
  def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, which_bn=bn, activation=None,
               upsample=None, skip_connection = True):
    super(GBlock2, self).__init__()

    self.in_channels, self.out_channels = in_channels, out_channels
    self.which_conv, self.which_bn = which_conv, which_bn
    self.activation = activation
    self.upsample = upsample

    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.out_channels)
    self.conv2 = self.which_conv(self.out_channels, self.out_channels)
    self.learnable_sc = in_channels != out_channels or upsample
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels,
                                     kernel_size=1, padding=0)

    # upsample layers
    self.upsample = upsample
    self.skip_connection = skip_connection

  def forward(self, x):
    h = self.activation(x)
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    h = self.conv1(h)
    #print(h.size())
    h = self.activation(h)
    h = self.conv2(h)
    # may be changed to h = self.conv2.forward_wo_sn(h)
    if self.learnable_sc:
      x = self.conv_sc(x)


    if self.skip_connection:
        out = h + x
    else:
        out = h
    return out


# Residual block for the discriminator
class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
               preactivation=False, activation=None, downsample=None,):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels if wide else self.in_channels
    self.which_conv = which_conv
    self.preactivation = preactivation
    self.activation = activation
    self.downsample = downsample

    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
    self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
    self.learnable_sc = True if (in_channels != out_channels) or downsample else False
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels,
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.preactivation:
      if self.learnable_sc:
        x = self.conv_sc(x)
      if self.downsample:
        x = self.downsample(x)
    else:
      if self.downsample:
        x = self.downsample(x)
      if self.learnable_sc:
        x = self.conv_sc(x)
    return x

  def forward(self, x):
    if self.preactivation:
      # h = self.activation(x) # NOT TODAY SATAN
      # Andy's note: This line *must* be an out-of-place ReLU or it
      #              will negatively affect the shortcut connection.
      h = F.relu(x)
    else:
      h = x
    h = self.conv1(h)
    h = self.conv2(self.activation(h))
    if self.downsample:
      h = self.downsample(h)

    return h + self.shortcut(x)


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
    arch = {}

    arch[256] = {'in_channels' :    [ch * item for item in [16, 16, 8, 8, 4, 2]],
                             'out_channels' : [ch * item for item in [16,    8, 8, 4, 2, 1]],
                             'upsample' : [True] * 6,
                             'resolution' : [8, 16, 32, 64, 128, 256],
                             'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                                            for i in range(3,9)}}
    arch[128] = {'in_channels' :    [ch * item for item in [16, 16, 8, 4, 2]],
                             'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
                             'upsample' : [True] * 5,
                             'resolution' : [8, 16, 32, 64, 128],
                             'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                                                            for i in range(3,8)}}
    return arch

class Generator(nn.Module):
    def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
                             G_kernel_size=3, G_attn='64', n_classes=2,
                             num_G_SVs=1, num_G_SV_itrs=1,
                             G_shared=True, shared_dim=0, hier=False,
                             cross_replica=False, mybn=False,
                             G_activation=nn.ReLU(inplace=False),
                             G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
                             BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
                             G_init='ortho', skip_init=False, no_optim=False,
                             G_param='SN', norm_style='bn',
                             **kwargs):
        super(Generator, self).__init__()
        # Channel width mulitplier
        self.ch = G_ch
        # Dimensionality of the latent space
        self.dim_z = dim_z
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_classes
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Architecture dict
        self.arch = G_arch(self.ch, self.attention)[resolution]

        self.unconditional = False

        # If using hierarchical latents, adjust z
        if self.hier:
            # Number of places z slots into
            self.num_slots = len(self.arch['in_channels']) + 1
            self.z_chunk_size = (self.dim_z // self.num_slots)

            if not self.unconditional:
                self.dim_z = self.z_chunk_size *    self.num_slots
        else:
            self.num_slots = 1
            self.z_chunk_size = 0

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        self.which_embedding = nn.Embedding

        if self.unconditional:
            bn_linear = nn.Linear
            input_size =  self.dim_z  + (self.shared_dim if self.G_shared else 0 )
        else:
            bn_linear = (functools.partial(self.which_linear, bias = False) if self.G_shared
                                     else self.which_embedding)

            input_size = (self.shared_dim + self.z_chunk_size if self.G_shared
                                    else self.n_classes)
        self.which_bn = functools.partial(ccbn,
                                                    which_linear=bn_linear,
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn,
                                                    input_size=input_size,
                                                    norm_style=self.norm_style,
                                                    eps=self.BN_eps,
                                                    self_modulation = self.unconditional)


        # Prepare model
        # If not using shared embeddings, self.shared is just a passthrough
        self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared
                                        else identity())
        # First linear layer
        if self.unconditional:
            self.linear = self.which_linear(self.dim_z, self.arch['in_channels'][0] * (self.bottom_width **2))
        else:
            self.linear = self.which_linear(self.dim_z // self.num_slots,
                                                                        self.arch['in_channels'][0] * (self.bottom_width **2))

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):


            self.blocks += [[GBlock(in_channels=self.arch['in_channels'][index],
                                                     out_channels=self.arch['out_channels'][index],
                                                     which_conv=self.which_conv,
						                             which_bn=self.which_bn,
                                                     activation=self.activation,
                                                     upsample=(functools.partial(F.interpolate, scale_factor=2)
                                                                         if self.arch['upsample'][index] else None))]]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(bn(self.arch['out_channels'][-1],
                                                                                                cross_replica=self.cross_replica,
                                                                                                mybn=self.mybn),
                                                                        self.activation,
                                                                        self.which_conv(self.arch['out_channels'][-1], 3))

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        # If this is an EMA copy, no need for an optim, so just return now
        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            print('Using fp16 adam in G...')
            self.optim = Adam16(params=self.parameters(), lr=self.lr,
                                                     betas=(self.B1, self.B2), weight_decay=0,
                                                     eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                                     betas=(self.B1, self.B2), weight_decay=0,
                                                     eps=self.adam_eps)

        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    def forward(self, z, y):
        # If hierarchical, concatenate zs and ys
        if self.hier:
            # faces
            if self.unconditional:
                ys = [z for _ in range(self.num_slots)]
            else:
                zs = torch.split(z, self.z_chunk_size, 1)
                z = zs[0]

                ys = [torch.cat([y, item], 1) for item in zs[1:]]
        else:
            if self.unconditional:
                ys = [None] * len(self.blocks)
            else:
                ys = [y] * len(self.blocks)

        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

        # Loop over blocks
        for index, blocklist in enumerate(self.blocks):
            # Second inner loop in case block has multiple layers
            for block in blocklist:
                h = block(h, ys[index])

        # Apply batchnorm-relu-conv-tanh at output
        return torch.tanh(self.output_layer(h))


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64',ksize='333333', dilation='111111'):
    arch = {}
    arch[256] = {'in_channels' :    [3] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
                             'downsample' : [True] * 6 + [False],
                             'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,8)}}
    arch[128] = {'in_channels' :    [3] + [ch*item for item in [1, 2, 4, 8, 16]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
                             'downsample' : [True] * 5 + [False],
                             'resolution' : [64, 32, 16, 8, 4, 4],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,8)}}
    return arch

def D_unet_arch(ch=64, attention='64',ksize='333333', dilation='111111',out_channel_multiplier=1):
    arch = {}

    n = 2

    ocm = out_channel_multiplier

    # covers bigger perceptual fields
    arch[128]= {'in_channels' :       [3] + [ch*item for item in       [1, 2, 4, 8, 16, 8*n, 4*2, 2*2, 1*2,1]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 8,   4,   2,    1,  1]],
                             'downsample' : [True]*5 + [False]*5,
                             'upsample':    [False]*5+ [True] *5,
                             'resolution' : [64, 32, 16, 8, 4, 8, 16, 32, 64, 128],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,11)}}


    arch[256] = {'in_channels' :            [3] + [ch*item for item in [1, 2, 4, 8, 8, 16, 8*2, 8*2, 4*2, 2*2, 1*2  , 1         ]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 8,   8,   4,   2,   1,   1          ]],
                             'downsample' : [True] *6 + [False]*6 ,
                             'upsample':    [False]*6 + [True] *6,
                             'resolution' : [128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256 ],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,13)}}



    return arch


class Unet_Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                             D_kernel_size=3, D_attn='64', n_classes=1000,
                             num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                             D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                             SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
                             D_init='ortho', skip_init=False, D_param='SN', decoder_skip_connection = True, **kwargs):
        super(Unet_Discriminator, self).__init__()


        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16



        if self.resolution==128:
            self.save_features = [0,1,2,3,4]
        elif self.resolution==256:
            self.save_features = [0,1,2,3,4,5]

        self.out_channel_multiplier = 1#4
        # Architecture
        self.arch = D_unet_arch(self.ch, self.attention , out_channel_multiplier = self.out_channel_multiplier  )[resolution]

        self.unconditional = False

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=self.SN_eps)

            self.which_embedding = functools.partial(SNEmbedding,
                                                            num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                            eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []

        for index in range(len(self.arch['out_channels'])):

            if self.arch["downsample"][index]:
                self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                             out_channels=self.arch['out_channels'][index],
                                             which_conv=self.which_conv,
                                             wide=self.D_wide,
                                             activation=self.activation,
                                             preactivation=(index > 0),
                                             downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

            elif self.arch["upsample"][index]:
                upsample_function = (functools.partial(F.interpolate, scale_factor=2, mode="nearest") #mode=nearest is default
                                    if self.arch['upsample'][index] else None)

                self.blocks += [[GBlock2(in_channels=self.arch['in_channels'][index],
                                                         out_channels=self.arch['out_channels'][index],
                                                         which_conv=self.which_conv,
                                                         #which_bn=self.which_bn,
                                                         activation=self.activation,
                                                         upsample= upsample_function, skip_connection = True )]]

            # If attention on this block, attach it to the end
            attention_condition = index < 5
            if self.arch['attention'][self.arch['resolution'][index]] and attention_condition: #index < 5
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                print("index = ", index)
                self.blocks[-1] += [Attention(self.arch['out_channels'][index],
                                                                                         self.which_conv)]


        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])


        last_layer = nn.Conv2d(self.ch*self.out_channel_multiplier,1,kernel_size=1)
        self.blocks.append(last_layer)
        #
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)

        self.linear_middle = self.which_linear(16*self.ch, output_dim)
        # Embedding for projection discrimination
        #if not kwargs["agnostic_unet"] and not kwargs["unconditional"]:
        #    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1]+extra)
        self.embed_middle = self.which_embedding(self.n_classes, 16*self.ch)
        self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])

        # Initialize weights
        if not skip_init:
            self.init_weights()

        ###
        print("_____params______")
        for name, param in self.named_parameters():
            print(name, param.size())

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            print('Using fp16 adam in D...')
            self.optim = Adam16(params=self.parameters(), lr=self.lr,
                                                         betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                                         betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)



    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x

        residual_features = []
        residual_features.append(x)
        # Loop over blocks

        for index, blocklist in enumerate(self.blocks[:-1]):
            if self.resolution == 128:
                if index==6 :
                    h = torch.cat((h,residual_features[4]),dim=1)
                elif index==7:
                    h = torch.cat((h,residual_features[3]),dim=1)
                elif index==8:#
                    h = torch.cat((h,residual_features[2]),dim=1)
                elif index==9:#
                    h = torch.cat((h,residual_features[1]),dim=1)

            if self.resolution == 256:
                if index==7:
                    h = torch.cat((h,residual_features[5]),dim=1)
                elif index==8:
                    h = torch.cat((h,residual_features[4]),dim=1)
                elif index==9:#
                    h = torch.cat((h,residual_features[3]),dim=1)
                elif index==10:#
                    h = torch.cat((h,residual_features[2]),dim=1)
                elif index==11:
                    h = torch.cat((h,residual_features[1]),dim=1)

            for block in blocklist:
                h = block(h)

            if index in self.save_features[:-1]:
                residual_features.append(h)

            if index==self.save_features[-1]:
                # Apply global sum pooling as in SN-GAN
                h_ = torch.sum(self.activation(h), [2, 3])
                # Get initial class-unconditional output
                bottleneck_out = self.linear_middle(h_)
                # Get projection of final featureset onto class vectors and add to evidence
                if self.unconditional:
                    projection = 0
                else:
                    # this is the bottleneck classifier c
                    emb_mid = self.embed_middle(y)
                    projection = torch.sum(emb_mid * h_, 1, keepdim=True)
                bottleneck_out = bottleneck_out + projection

        out = self.blocks[-1](h)

        if self.unconditional:
            proj = 0
        else:
            emb = self.embed(y)
            emb = emb.view(emb.size(0),emb.size(1),1,1).expand_as(h)
            proj = torch.sum(emb * h, 1, keepdim=True)
            ################
        out = out + proj

        out = out.view(out.size(0),1,self.resolution,self.resolution)

        return out, bottleneck_out


# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):

    def __init__(self, G, D, config):
        super(G_D, self).__init__()
        self.G = G
        self.D = D

        self.config = config

    def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
                            split_D=False, dw1=[],dw2=[], reference_x = None, mixup = False, mixup_only = False, target_map=None):

        if mixup:
            gy = dy
            #why? so the mixup samples consist of same class

        # If training G, enable grad tape
        with torch.set_grad_enabled(train_G):

            G_z = self.G(z, self.G.shared(gy))
            # Cast as necessary
            if self.G.fp16 and not self.D.fp16:
                G_z = G_z.float()
            if self.D.fp16 and not self.G.fp16:
                G_z = G_z.half()

        if mixup:
            initial_x_size = x.size(0)

            mixed = target_map*x+(1-target_map)*G_z
            mixed_y = dy


        if not mixup_only:
            # we get here in the cutmix cons extra case
            D_input = torch.cat([G_z, x], 0) if x is not None else G_z
            D_class = torch.cat([gy, dy], 0) if dy is not None else gy
            dmap = torch.tensor([])
            if mixup:
                #we get here in the cutmix  "consistency loss and augmentation" case, if "mixup" is true for the current round (depends on p mixup)
                D_input = torch.cat([D_input, mixed], 0)
                if self.config["dataset"]!="coco_animals":
                    D_class = torch.cat([D_class.float(), mixed_y.float()], 0)
                else:
                    D_class = torch.cat([D_class.long(), mixed_y.long()], 0)
        else:
            #not reached in cutmix "consistency loss and augmentation"
            D_input = mixed
            D_class = mixed_y
            dmap = torch.tensor([])

            del G_z
            del x
            G_z = None
            x = None

        D_out, D_middle = self.D(D_input, D_class)

        del D_input
        del D_class


        if x is not None:

            if not mixup:
                out = torch.split(D_out, [G_z.shape[0], x.shape[0]])     # D_fake, D_real
            else:
                out = torch.split(D_out, [G_z.shape[0], x.shape[0], mixed.shape[0]])  # D_fake, D_real, D_mixed
            out = out + (G_z,)
            if mixup:
                out = out + (mixed,)

            if not mixup:
                D_middle =  torch.split(D_middle, [G_z.shape[0], x.shape[0]])     # D_middle_fake, D_middle_real
            else:
                D_middle =  torch.split(D_middle, [G_z.shape[0], x.shape[0] , mixed.shape[0]])
            out = out + D_middle
            ###return target map as well
            if mixup:
                out = out + (target_map,)

            return out


        else:
            #in mixup# you arrive here
            out = (D_out,)

            if return_G_z:
                out = out + (G_z,)
            if mixup_only:
                out = out + (mixed,)

            out =  out + (D_middle,)
            ##return target map as well
            if mixup:
                out = out + (target_map,)

            return out
        


if __name__ == '__main__':
   g = Generator()
   t = torch.rand(1, 64, 128, 128)
   res = g(t)
   print(res.shape)