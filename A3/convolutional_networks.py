"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
from eecs598 import Solver
from a3_helper import svm_loss, softmax_loss
from fully_connected_networks import *
import itertools
from math import sqrt

def hello_convolutional_networks():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from convolutional_networks.py!')


class Conv(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
      
    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ##############################################################################
    # TODO: Implement the convolutional forward pass.                            #
    # Hint: you can use the function torch.nn.functional.pad for padding.        #
    # Note that you are NOT allowed to use anything in torch.nn in other places. #
    ##############################################################################
    # Replace "pass" statement with your code
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    device = x.device
    dtype = x.dtype
    
    # vertical padding
    padded_x = torch.cat([torch.zeros((N, C, pad, x.size(3)), device=device, dtype=dtype), 
                         x, 
                         torch.zeros((N, C, pad, x.size(3)), device=device, dtype=dtype),
                        ],
                        dim=2)

    # horizontal padding
    padded_x = torch.cat([torch.zeros((N, C, x.size(2) + 2*pad, pad), device=device, dtype=dtype), 
                         padded_x, 
                         torch.zeros((N, C, x.size(2) + 2*pad, pad), device=device, dtype=dtype),
                        ],
                        dim=3)

    padded_h = x.size(2) + 2*pad
    padded_w = x.size(3) + 2*pad
    H_out = int(1 + (padded_h - HH) / stride)
    W_out = int(1 + (padded_w - WW) / stride)
    out = torch.zeros((N, F, H_out, W_out), device=device, dtype=dtype)

    # iterating over filter location
    filters = w.flatten(start_dim=1)
    for i, filter_x in enumerate(range(0, padded_h - HH + 1, stride)):
      for j, filter_y in enumerate(range(0, padded_w - WW + 1, stride)):
        patch = padded_x[:, :, filter_x:filter_x + WW, filter_y:filter_y + HH].flatten(start_dim=1)
        out[:, :, i, j] = patch@filters.t() + b
        

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    # Replace "pass" statement with your code
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    device = x.device
    dtype = x.dtype

    padded_h = x.size(2) + 2*pad
    padded_w = x.size(3) + 2*pad
    H_out = int(1 + (padded_h - HH) / stride)
    W_out = int(1 + (padded_w - WW) / stride)

    # vertical padding
    padded_x = torch.cat([torch.zeros((N, C, pad, x.size(3)), device=device, dtype=dtype), 
                         x, 
                         torch.zeros((N, C, pad, x.size(3)), device=device, dtype=dtype),
                        ],
                        dim=2)

    # horizontal padding
    padded_x = torch.cat([torch.zeros((N, C, x.size(2) + 2*pad, pad), device=device, dtype=dtype), 
                         padded_x, 
                         torch.zeros((N, C, x.size(2) + 2*pad, pad), device=device, dtype=dtype),
                        ],
                        dim=3)

    dx = torch.zeros((N, C, H, W), device=device, dtype=dtype)
    dw = torch.zeros_like(w, device=device, dtype=dtype)
    db = torch.zeros_like(b, device=device, dtype=dtype)

    # db
    db = dout.sum(dim=[0, 2, 3])

    # dw
    for f in range(F):
      for c in range(C):
        for i, filter_x in enumerate(range(0, padded_h - HH + 1, stride)):
          for j, filter_y in enumerate(range(0, padded_w - WW + 1, stride)):
            # for each input channel c and output channel f
            # we calculate the effect of the weights on the current patch
            # i.e. patch of the size of the filter * the respective dout entry
            x_patch = padded_x[:, c, filter_x: filter_x + HH, filter_y: filter_y + WW]
            dout_entry = dout[:, f, i, j]
            dw[f, c, :, :] += (x_patch.transpose(0, 2) * dout_entry).transpose(0, 2).sum(dim=0)

    # dx
    # dilating the upstream gradient
    H_dilated = stride * (H_out-1) + 1
    W_dilated = stride * (W_out-1) + 1
    dilated_dout = torch.zeros((N, F, H_dilated, W_dilated), device=device, dtype=dtype)

    # inserting dout in the proper entries in the sparse tensor 'dilated_dout'
    indices_product = itertools.product(range(0, H_dilated, stride), range(0, W_dilated, stride))
    indices = torch.tensor(list(indices_product)).t()
    dilated_dout[:, :, indices[0], indices[1]] = dout.view((N, F, -1))


    # dout padding to obtain the correct shape of the downstream grad
    dout_pad_h = max(HH - pad - 1, 0)
    dout_pad_w = max(WW - pad - 1, 0)
    # vertical padding
    padded_dout = torch.cat([torch.zeros((N, F, dout_pad_h, W_dilated), device=device, dtype=dtype), 
                         dilated_dout, 
                         torch.zeros((N, F, dout_pad_h, W_dilated), device=device, dtype=dtype),
                        ],
                        dim=2)

    # horizontal padding
    padded_dout = torch.cat([torch.zeros((N, F, H_dilated + 2*dout_pad_h, dout_pad_w), device=device, dtype=dtype), 
                         padded_dout, 
                         torch.zeros((N, F, H_dilated + 2*dout_pad_h, dout_pad_w), device=device, dtype=dtype),
                        ],
                        dim=3)

    padded_dout_h = H_dilated + 2*dout_pad_h
    padded_dout_w = W_dilated + 2*dout_pad_w


    for n in range(N):
      for c in range(C):
        for i, filter_x in enumerate(range(0, padded_dout_h - HH + 1)):
          for j, filter_y in enumerate(range(0, padded_dout_w - WW + 1)):
            dout_patch = padded_dout[n, :, filter_x: filter_x + HH, filter_y: filter_y + WW]
            w_channel = w[:, c].flip(dims=(1,2))   # double flip
            dx[n, c, i, j] = (dout_patch * w_channel).sum()

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return dx, dw, db


class MaxPool(object):

  @staticmethod
  def forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here.

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max-pooling forward pass                              #
    #############################################################################
    # Replace "pass" statement with your code
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    dtype = x.dtype
    device = x.device
    pooled_h = int(1 + (H - pool_height)/stride)
    pooled_w = int(1 + (W - pool_width)/stride)
    out = torch.zeros((N, C, pooled_h, pooled_w), dtype=dtype, device=device)

    for i, filter_x in enumerate(range(0, H - pool_height + 1, stride)):
      for j, filter_y in enumerate(range(0, W - pool_width + 1, stride)):
        patch = x[:, :, filter_x: filter_x + pool_height, filter_y: filter_y + pool_width]
        out[:, :, i, j], _ = patch.flatten(start_dim=2).max(dim=2)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    cache = (x, pool_param)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max-pooling backward pass                             #
    #############################################################################
    # Replace "pass" statement with your code
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    dtype = x.dtype
    device = x.device
    dx = torch.zeros_like(x, dtype=dtype, device=device)

    for i, filter_x in enumerate(range(0, H - pool_height + 1, stride)):
      for j, filter_y in enumerate(range(0, W - pool_width + 1, stride)):
        patch = x[:, :, filter_x: filter_x + pool_height, filter_y: filter_y + pool_width]
        # take max per patch
        max_vals, _ = patch.flatten(start_dim=2).max(dim=2)
        # get a tensor of ones at the entry of the max value, per patch
        bool_max_vals = (patch.flatten(start_dim=2).transpose(0, 2) == max_vals.t()).to(dtype)
        # multiplying by the upstream grad, transpose to adjust the shape for the broadcast multiplication
        downstream_max_vals = bool_max_vals * dout[:, :, i, j].t()
        # transposing back to (N, C, flattened_patch) and reshaping back to (N, C, patch_height, patch_width)
        local_downstream_grad = downstream_max_vals.transpose(0, 2).reshape(patch.shape)
        dx[:, :, filter_x: filter_x + pool_height, filter_y: filter_y + pool_width] = local_downstream_grad

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return dx


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  conv - relu - 2x2 max pool - linear - relu - linear - softmax
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dims=(3, 32, 32), num_filters=32, filter_size=7,
         hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
         dtype=torch.float, device='cpu'):
    """
    Initialize a new network.
    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Width/height of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian centered at 0.0   #
    # with standard deviation equal to weight_scale; biases should be          #
    # initialized to zero. All weights and biases should be stored in the      #
    #  dictionary self.params. Store weights and biases for the convolutional  #
    # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
    # weights and biases of the hidden linear layer, and keys 'W3' and 'b3'    #
    # for the weights and biases of the output linear layer.                   #
    #                                                                          #
    # IMPORTANT: For this assignment, you can assume that the padding          #
    # and stride of the first convolutional layer are chosen so that           #
    # **the width and height of the input are preserved**. Take a look at      #
    # the start of the loss() function to see how that happens.                #               
    ############################################################################
    # Replace "pass" statement with your code
    self.device = device
    in_channels, H, W = input_dims
    fc_in_size = num_filters * (H//2) * (W//2)
    self.params['W1'] = torch.randn((num_filters, in_channels, filter_size, filter_size), dtype=dtype, device=device) * weight_scale
    self.params['b1'] = torch.zeros((num_filters,), dtype=dtype, device=device)
    self.params['W2'] = torch.randn((fc_in_size, hidden_dim), dtype=dtype, device=device) * weight_scale
    self.params['b2'] = torch.zeros((hidden_dim,), dtype=dtype, device=device)
    self.params['W3'] = torch.randn((hidden_dim, num_classes), dtype=dtype, device=device) * weight_scale
    self.params['b3'] = torch.zeros((num_classes,), dtype=dtype, device=device)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def save(self, path):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))

  def load(self, path):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.dtype = checkpoint['dtype']
    self.reg = checkpoint['reg']
    print("load checkpoint file: {}".format(path))


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    Input / output: Same API as TwoLayerNet.
    """
    X = X.to(self.dtype)
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    #                                                                          #
    # Remember you can use the functions defined in your implementation above. #
    ############################################################################
    # Replace "pass" statement with your code
    a1, conv_cache = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
    a2, fc1_cache = Linear_ReLU.forward(a1, W2, b2)
    z3, fc2_cache = Linear.forward(a2, W3, b3)
    scores = z3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization does not include  #
    # a factor of 0.5                                                          #
    ############################################################################
    # Replace "pass" statement with your code
    ce_loss, loss_grad = softmax_loss(scores, y)
    loss = ce_loss + self.reg * ((W1**2).sum() + (W2**2).sum() + (W3**2).sum())

    dx3, dw3, db3 = Linear.backward(loss_grad, fc2_cache)
    dx2, dw2, db2 = Linear_ReLU.backward(dx3, fc1_cache)
    dx1, dw1, db1 = Conv_ReLU_Pool.backward(dx2, conv_cache)

    grads['W1'] = dw1 + 2*self.reg*W1
    grads['b1'] = db1
    grads['W2'] = dw2 + 2*self.reg*W2
    grads['b2'] = db2
    grads['W3'] = dw3 + 2*self.reg*W3
    grads['b3'] = db3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

class DeepConvNet(object):
  """
  A convolutional neural network with an arbitrary number of convolutional
  layers in VGG-Net style. All convolution layers will use kernel size 3 and 
  padding 1 to preserve the feature map size, and all pooling layers will be
  max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
  size of the feature map.

  The network will have the following architecture:
  
  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

  Each {...} structure is a "macro layer" consisting of a convolution layer,
  an optional batch normalization layer, a ReLU nonlinearity, and an optional
  pooling layer. After L-1 such macro layers, a single fully-connected layer
  is used to predict the class scores.

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dims=(3, 32, 32),
               num_filters=[8, 8, 8, 8, 8],
               max_pools=[0, 1, 2, 3, 4],
               batchnorm=False,
               num_classes=10, weight_scale=1e-3, reg=0.0,
               weight_initializer=None,
               dtype=torch.float, device='cpu'):
    """
    Initialize a new network.

    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: List of length (L - 1) giving the number of convolutional
      filters to use in each macro layer.
    - max_pools: List of integers giving the indices of the macro layers that
      should have max pooling (zero-indexed).
    - batchnorm: Whether to include batch normalization in each macro layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights, or the string "kaiming" to use Kaiming initialization instead
    - reg: Scalar giving L2 regularization strength. L2 regularization should
      only be applied to convolutional and fully-connected weight matrices;
      it should not be applied to biases or to batchnorm scale and shifts.
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'    
    """
    self.params = {}
    self.num_layers = len(num_filters)+1
    self.max_pools = max_pools
    self.batchnorm = batchnorm
    self.reg = reg
    self.dtype = dtype
  
    if device == 'cuda':
      device = 'cuda:0'
    
    ############################################################################
    # TODO: Initialize the parameters for the DeepConvNet. All weights,        #
    # biases, and batchnorm scale and shift parameters should be stored in the #
    # dictionary self.params.                                                  #
    #                                                                          #
    # Weights for conv and fully-connected layers should be initialized        #
    # according to weight_scale. Biases should be initialized to zero.         #
    # Batchnorm scale (gamma) and shift (beta) parameters should be initilized #
    # to ones and zeros respectively.                                          #           
    ############################################################################
    # Replace "pass" statement with your code
    in_channels, H, W = input_dims
    for i in range(self.num_layers - 1):
      if i > 0:
        in_channels = num_filters[i - 1]

      self.params[f'b{i+1}'] = torch.zeros((num_filters[i],), dtype=dtype, device=device)
      if weight_scale == 'kaiming':
        self.params[f'W{i+1}'] = kaiming_initializer(in_channels, num_filters[i], K=3, dtype=dtype, device=device)
      else:
        self.params[f'W{i+1}'] = torch.randn((num_filters[i], in_channels, 3, 3), dtype=dtype, device=device) * weight_scale

      if batchnorm:
        self.params[f'gamma{i+1}'] = torch.ones((num_filters[i],), dtype=dtype, device=device)
        self.params[f'beta{i+1}'] = torch.zeros((num_filters[i],), dtype=dtype, device=device)


    conv_out_num_filters = num_filters[-1]
    conv_out_H = int(H / (2**len(max_pools)))
    conv_out_W = int(W / (2**len(max_pools)))
    fc_in_size = conv_out_num_filters * conv_out_H * conv_out_W

    self.params[f'b{self.num_layers}'] = torch.zeros((num_classes, ), dtype=dtype, device=device)
    if weight_scale == 'kaiming':
      self.params[f'W{self.num_layers}'] = kaiming_initializer(fc_in_size, num_classes, dtype=dtype, device=device)
    else:
      self.params[f'W{self.num_layers}'] = torch.randn((fc_in_size, num_classes), dtype=dtype, device=device) * weight_scale

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.batchnorm:
      self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]
      
    # Check that we got the right number of parameters
    if not self.batchnorm:
      params_per_macro_layer = 2  # weight and bias
    else:
      params_per_macro_layer = 4  # weight, bias, scale, shift
    num_params = params_per_macro_layer * len(num_filters) + 2
    msg = 'self.params has the wrong number of elements. Got %d; expected %d'
    msg = msg % (len(self.params), num_params)
    assert len(self.params) == num_params, msg

    # Check that all parameters have the correct device and dtype:
    for k, param in self.params.items():
      msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)
      assert param.device == torch.device(device), msg
      msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)
      assert param.dtype == dtype, msg


  def save(self, path):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
      'num_layers': self.num_layers,
      'max_pools': self.max_pools,
      'batchnorm': self.batchnorm,
      'bn_params': self.bn_params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))


  def load(self, path, dtype, device):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.dtype = dtype
    self.reg = checkpoint['reg']
    self.num_layers = checkpoint['num_layers']
    self.max_pools = checkpoint['max_pools']
    self.batchnorm = checkpoint['batchnorm']
    self.bn_params = checkpoint['bn_params']


    for p in self.params:
      self.params[p] = self.params[p].type(dtype).to(device)

    for i in range(len(self.bn_params)):
      for p in ["running_mean", "running_var"]:
        self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)

    print("load checkpoint file: {}".format(path))


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the deep convolutional network.
    Input / output: Same API as ThreeLayerConvNet.
    """
    X = X.to(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params since they
    # behave differently during training and testing.
    if self.batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode
    scores = None

    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = 3
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the DeepConvNet, computing the      #
    # class scores for X and storing them in the scores variable.              #
    #                                                                          #
    # You should use the fast versions of convolution and max pooling layers,  #
    # or the convolutional sandwich layers, to simplify your implementation.   #
    ############################################################################
    # Replace "pass" statement with your code
    caches = []
    previous_out = X
    for i in range(self.num_layers - 1):
      w_i = self.params[f'W{i+1}']
      b_i = self.params[f'b{i+1}']
      if self.batchnorm:
        gamma_i = self.params[f'gamma{i+1}']
        beta_i = self.params[f'beta{i+1}']
        bn_param_i = self.bn_params[i]

      if i not in self.max_pools and not self.batchnorm:
        previous_out, conv_cache = Conv_ReLU.forward(previous_out, w_i, b_i, conv_param)

      elif i in self.max_pools and not self.batchnorm:
        previous_out, conv_cache = Conv_ReLU_Pool.forward(previous_out, w_i, b_i, conv_param, pool_param)

      elif i not in self.max_pools and self.batchnorm:
        previous_out, conv_cache = Conv_BatchNorm_ReLU.forward(previous_out, w_i, b_i, gamma_i, beta_i, conv_param, bn_param_i)

      elif i in self.max_pools and self.batchnorm:
        previous_out, conv_cache = Conv_BatchNorm_ReLU_Pool.forward(previous_out, w_i, b_i, gamma_i, beta_i, conv_param, bn_param_i, pool_param)
      
      caches.append(conv_cache)

    fc_w = self.params[f'W{self.num_layers}']
    fc_b = self.params[f'b{self.num_layers}']
    scores, fc_cache = Linear.forward(previous_out, fc_w, fc_b)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the DeepConvNet, storing the loss  #
    # and gradients in the loss and grads variables. Compute data loss using   #
    # softmax, and make sure that grads[k] holds the gradients for             #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization does not include  #
    # a factor of 0.5                                                          #
    ############################################################################
    # Replace "pass" statement with your code
    ce_loss, loss_grad = softmax_loss(scores, y)
    loss += ce_loss
    dx, dw, db = Linear.backward(loss_grad, fc_cache)
    grads[f'W{self.num_layers}'] = dw + 2*self.reg*self.params[f'W{self.num_layers}']
    grads[f'b{self.num_layers}'] = db
    loss += self.reg * (self.params[f'W{self.num_layers}']**2).sum()

    for i in range(self.num_layers - 2, -1, -1):
      if i not in self.max_pools and not self.batchnorm:
        dx, dw, db = Conv_ReLU.backward(dx, caches[i])

      if i in self.max_pools and not self.batchnorm:
        dx, dw, db = Conv_ReLU_Pool.backward(dx, caches[i])

      if i not in self.max_pools and self.batchnorm:
        dx, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(dx, caches[i])

      if i in self.max_pools and self.batchnorm:
        dx, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU_Pool.backward(dx, caches[i])

      grads[f'W{i + 1}'] = dw + 2*self.reg*self.params[f'W{i + 1}']
      grads[f'b{i + 1}'] = db
      if self.batchnorm:
        grads[f'gamma{i + 1}'] = dgamma
        grads[f'beta{i + 1}'] = dbeta
      loss += self.reg * (self.params[f'W{i + 1}']**2).sum()
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

def find_overfit_parameters():
  weight_scale = 2e-3   # Experiment with this!
  learning_rate = 1e-5  # Experiment with this!
  ############################################################################
  # TODO: Change weight_scale and learning_rate so your model achieves 100%  #
  # training accuracy within 30 epochs.                                      #
  ############################################################################
  # Replace "pass" statement with your code
  weight_scale = 10
  learning_rate = 0.1
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################
  return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
  model = None
  solver = None
  ################################################################################
  # TODO: Train the best DeepConvNet that you can on CIFAR-10 within 60 seconds. #
  ################################################################################
  # Replace "pass" statement with your code

  X_train, y_train = data_dict['X_train'], data_dict['y_train']
  input_dims = X_train.shape[1:]
  dtype = X_train.dtype
  device = X_train.device

  model = DeepConvNet(input_dims=input_dims,
                      num_filters=[32, 64, 96],
                      max_pools=[0, 1],
                      batchnorm=False,
                      num_classes=10,
                      weight_scale='kaiming',
                      reg=1e-5,
                      dtype=dtype,
                      device=device)

  num_epochs = 4
  batch_size = 128
  lr = 1e-3
  solver = Solver(model, data_dict,
                num_epochs=num_epochs, batch_size=batch_size,
                update_rule=adam,
                optim_config={
                  'learning_rate': lr,
                },
                verbose=False, device='cuda')

  ################################################################################
  #                              END OF YOUR CODE                                #
  ################################################################################
  return solver

def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
  """
  Implement Kaiming initialization for linear and convolution layers.
  
  Inputs:
  - Din, Dout: Integers giving the number of input and output dimensions for
    this layer
  - K: If K is None, then initialize weights for a linear layer with Din input
    dimensions and Dout output dimensions. Otherwise if K is a nonnegative
    integer then initialize the weights for a convolution layer with Din input
    channels, Dout output channels, and a kernel size of KxK.
  - relu: If ReLU=True, then initialize weights with a gain of 2 to account for
    a ReLU nonlinearity (Kaiming initializaiton); otherwise initialize weights
    with a gain of 1 (Xavier initialization).
  - device, dtype: The device and datatype for the output tensor.

  Returns:
  - weight: A torch Tensor giving initialized weights for this layer. For a
    linear layer it should have shape (Din, Dout); for a convolution layer it
    should have shape (Dout, Din, K, K).
  """
  gain = 2. if relu else 1.
  weight = None
  if K is None:
    ###########################################################################
    # TODO: Implement Kaiming initialization for linear layer.                #
    # The weight scale is sqrt(gain / fan_in),                                #
    # where gain is 2 if ReLU is followed by the layer, or 1 if not,          #
    # and fan_in = num_in_channels (= Din).                                   #
    # The output should be a tensor in the designated size, dtype, and device.#
    ###########################################################################
    # Replace "pass" statement with your code
    weight_scale = sqrt(gain / Din)
    weight = torch.normal(0, weight_scale, (Din, Dout), dtype=dtype, device=device)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  else:
    ###########################################################################
    # TODO: Implement Kaiming initialization for convolutional layer.         #
    # The weight scale is sqrt(gain / fan_in),                                #
    # where gain is 2 if ReLU is followed by the layer, or 1 if not,          #
    # and fan_in = num_in_channels (= Din) * K * K                            #
    # The output should be a tensor in the designated size, dtype, and device.#
    ###########################################################################
    # Replace "pass" statement with your code
    fan_in = Din * (K**2)
    weight_scale = sqrt(gain / fan_in)
    weight = torch.normal(0, weight_scale, (Dout, Din, K, K), dtype=dtype, device=device)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  return weight

class BatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the PyTorch
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
    running_var = bn_param.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))

    out, cache = None, None
    if mode == 'train':
      #######################################################################
      # TODO: Implement the training-time forward pass for batch norm.      #
      # Use minibatch statistics to compute the mean and variance, use      #
      # these statistics to normalize the incoming data, and scale and      #
      # shift the normalized data using gamma and beta.                     #
      #                                                                     #
      # You should store the output in the variable out. Any intermediates  #
      # that you need for the backward pass should be stored in the cache   #
      # variable.                                                           #
      #                                                                     #
      # You should also use your computed sample mean and variance together #
      # with the momentum variable to update the running mean and running   #
      # variance, storing your result in the running_mean and running_var   #
      # variables.                                                          #
      #                                                                     #
      # Note that though you should be keeping track of the running         #
      # variance, you should normalize the data based on the standard       #
      # deviation (square root of variance) instead!                        # 
      # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
      # might prove to be helpful.                                          #
      #######################################################################
      # Replace "pass" statement with your code
      batch_mean = x.mean(dim=0)
      batch_var = x.var(dim=0)
      normalized_x = (x - batch_mean) / (torch.sqrt(batch_var + eps))
      out = normalized_x*gamma + beta
      
      running_mean = momentum*running_mean + (1-momentum)*batch_mean 
      running_var = momentum*running_var + (1-momentum)*batch_var 

      cache = (x, gamma, beta, batch_mean, batch_var, mode, eps)

      #######################################################################
      #                           END OF YOUR CODE                          #
      #######################################################################
    elif mode == 'test':
      #######################################################################
      # TODO: Implement the test-time forward pass for batch normalization. #
      # Use the running mean and variance to normalize the incoming data,   #
      # then scale and shift the normalized data using gamma and beta.      #
      # Store the result in the out variable.                               #
      #######################################################################
      # Replace "pass" statement with your code
      normalized_x = (x - running_mean) / (torch.sqrt(running_var + eps))
      out = normalized_x*gamma + beta
      #######################################################################
      #                           END OF YOUR CODE                          #
      #######################################################################
    else:
      raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean.detach()
    bn_param['running_var'] = running_var.detach()

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    # Don't forget to implement train and test mode separately.               #
    ###########################################################################
    # Replace "pass" statement with your code
    x, gamma, beta, batch_mean, batch_var, mode, eps = cache
    N, D = x.shape

    if mode == 'train':
      dbeta = dout.sum(dim=0)

      normalized_x = (x - batch_mean) / (torch.sqrt(batch_var + eps))
      dgamma = (normalized_x * dout).sum(dim=0)

      dxhat = dout * gamma
      dvar = -(1/2) * ((batch_var+eps)**(-3/2)) * ((x-batch_mean) * dxhat).sum(dim=0)
      dmean = -(1/torch.sqrt(batch_var+eps)) * dxhat.sum(dim=0)
      dx = (1/N) * dmean \
           + (2/N) * (x-batch_mean) * dvar  \
           + (1/torch.sqrt(batch_var+eps)) * dxhat


    elif mode == 'test':
      pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

  @staticmethod
  def backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
    
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # Replace "pass" statement with your code
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


class SpatialBatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # Replace "pass" statement with your code
    N, C, H, W = x.shape
    x_flat = x.transpose(1, 3).reshape((N*H*W, C))
    flat_out, cache = BatchNorm.forward(x_flat, gamma, beta, bn_param)
    out = flat_out.reshape((N, W, H, C)).transpose(1, 3)
    cache = (cache, x.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # Replace "pass" statement with your code
    cache, (N, C, H, W) = cache
    dout_flat = dout.transpose(1, 3).reshape((N*H*W, C))
    dx_flat, dgamma, dbeta = BatchNorm.backward(dout_flat, cache)
    dx = dx_flat.reshape((N, W, H, C)).transpose(1, 3)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


################################################################################
################################################################################
#################   Fast Implementations and Sandwich Layers  ##################
################################################################################
################################################################################

class FastConv(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
    layer.weight = torch.nn.Parameter(w)
    layer.bias = torch.nn.Parameter(b)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, w, b, conv_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, _, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
      dw = layer.weight.grad.detach()
      db = layer.bias.grad.detach()
      layer.weight.grad = layer.bias.grad = None
    except RuntimeError:
      dx, dw, db = torch.zeros_like(tx), torch.zeros_like(layer.weight), torch.zeros_like(layer.bias)
    return dx, dw, db


class FastMaxPool(object):

  @staticmethod
  def forward(x, pool_param):
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=stride)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, pool_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
    except RuntimeError:
      dx = torch.zeros_like(tx)
    return dx

class Conv_ReLU(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    out, relu_cache = ReLU.forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = ReLU.backward(dout, relu_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db


class Conv_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, b, conv_param, pool_param):
    """
    A convenience layer that performs a convolution, a ReLU, and a pool.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    s, relu_cache = ReLU.forward(a)
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = FastMaxPool.backward(dout, pool_cache)
    da = ReLU.backward(ds, relu_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db

class Linear_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an linear transform, batch normalization,
    and ReLU.
    Inputs:
    - x: Array of shape (N, D1); input to the linear layer
    - w, b: Arrays of shape (D2, D2) and (D2,) giving the weight and bias for
      the linear transform.
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.
    Returns:
    - out: Output from ReLU, of shape (N, D2)
    - cache: Object to give to the backward pass.
    """
    a, fc_cache = Linear.forward(x, w, b)
    a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
    out, relu_cache = ReLU.forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the linear-batchnorm-relu convenience layer.
    """
    fc_cache, bn_cache, relu_cache = cache
    da_bn = ReLU.backward(dout, relu_cache)
    da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
    dx, dw, db = Linear.backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    out, relu_cache = ReLU.forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = ReLU.backward(dout, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    s, relu_cache = ReLU.forward(an)
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = FastMaxPool.backward(dout, pool_cache)
    dan = ReLU.backward(ds, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db, dgamma, dbeta
