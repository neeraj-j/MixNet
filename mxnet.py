# This is pytorch mplementation of mixnet from
# https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import torch
import torch.nn as nn

import collections
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from typing import List

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'depth_multiplier', 'depth_divisor', 'min_depth',
    'use_keras', 'stem_size', 'feature_size',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'dw_ksize', 'expand_ksize', 'project_ksize', 'num_repeat', 'input_filters',
    'output_filters', 'expand_ratio', 'id_skip', 'strides', 'se_ratio',
    'swish', 'dilated',
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, bias=True, momentum=0.1, eps=1e-05, groups=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_planes, momentum=momentum, eps=eps),
            nn.ReLU(inplace=True)
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, bias=True, momentum=0.1, eps=1e-05,groups=1):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_planes, momentum=momentum, eps=eps),
        )

class ConvReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, bias=True, momentum=0.1, groups=1):
        super(ConvReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias, groups=groups),
            nn.ReLU(inplace=True)
        )

def _split_channels(total_filters, num_groups):
  split = [total_filters // num_groups for _ in range(num_groups)]
  split[0] += total_filters - sum(split)
  return split


class MixConv(nn.Module):
  """MixConv with mixed depthwise convolutional kernels.

  MDConv is an improved depthwise convolution that mixes multiple kernels (e.g.
  3x3, 5x5, etc). Right now, we use an naive implementation that split channels
  into multiple groups and perform different kernels for each group.

  See Mixnet paper for more details.
  """

  def __init__(self, channels, kernel_size, stride):
      super(MixConv, self).__init__()
      # in and out channels are always same
      self.q_cat = torch.nn.quantized.FloatFunctional()
      self.num_groups = len(kernel_size)
      self.split_channels = _split_channels(channels, self.num_groups)

      self.mixed_conv = nn.ModuleList()
      for i in range(self.num_groups):
          self.mixed_conv.append(nn.Conv2d(
              self.split_channels[i],
              self.split_channels[i],
              kernel_size[i],
              stride=stride,
              padding=kernel_size[i] // 2,
              groups=self.split_channels[i],
              bias=False
          ))

  def forward(self, x):
      x_split = torch.split(x, self.split_channels, dim=1)
      # x = [conv(t) for conv, t in zip(self.mixed_depthwise_conv, x_split)]
      x: List[torch.Tensor] = []
      for i, conv in enumerate(self.mixed_conv):
          x.append(conv(x_split[i]))
      x = self.q_cat.cat(x, dim=1)

      return x


def round_filters(filters, global_params):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.depth_multiplier
  divisor = global_params.depth_divisor
  min_depth = global_params.min_depth
  if not multiplier:
    return filters

  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return new_filters


class MixnetBlock(nn.Module):
  """A class of Mixnet block.

  Attributes:
    endpoints: dict. A list of internal tensors.
  """

  def __init__(self, block_args, global_params):
    """Initializes the block.

    Args:
      block_args: BlockArgs, arguments to create a MixnetBlock.
      global_params: GlobalParams, a set of global parameters.
    """
    super(MixnetBlock, self).__init__()
    self._block_args = block_args
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    self._use_keras = global_params.use_keras
    self._data_format = global_params.data_format

    self._channel_axis = 1
    self._spatial_dims = [2, 3]

    self._has_se = (self._block_args.se_ratio is not None) and (
        self._block_args.se_ratio > 0) and (self._block_args.se_ratio <= 1)
    self._relu_fn = nn.ReLU(inplace=True)

    """Builds block according to the arguments."""
    in_filters = self._block_args.input_filters
    out_filters = self._block_args.input_filters * self._block_args.expand_ratio
    if self._block_args.expand_ratio != 1:
      # Expansion phase:
      groups = len(self._block_args.expand_ksize)
      self._expand_conv = ConvBNReLU(in_filters, out_filters, 1, 1, 0, bias=False,
                                     momentum=self._batch_norm_momentum,
                                     eps=self._batch_norm_epsilon, groups=groups)

    kernel_size = self._block_args.dw_ksize
    # Depth-wise convolution phase:
    self._depthwise_conv = MixConv(
        out_filters,
        kernel_size,
        self._block_args.strides)
    self._bn1 = nn.BatchNorm2d(out_filters, momentum=self._batch_norm_momentum,
                               eps=self._batch_norm_epsilon)

    if self._has_se:
      num_reduced_filters = max(
          1, int(self._block_args.input_filters * self._block_args.se_ratio))
      # Squeeze and Excitation layer.
      self._se_reduce = ConvReLU(
          out_filters, num_reduced_filters,1, 1, padding=0, bias=True)
      self._se_expand = nn.Conv2d(
          num_reduced_filters, out_filters,1,1,padding=0, bias=True,)

    # Output phase:
    filters = self._block_args.output_filters
    groups = len(self._block_args.project_ksize)
    self._project_conv = ConvBN(
        out_filters, filters, 1, 1, padding=0,bias=False, groups=groups,
        momentum=self._batch_norm_momentum, eps=self._batch_norm_epsilon)

    self.residual = False
    if self._block_args.id_skip:
        if all(
                s == 1 for s in self._block_args.strides
        ) and self._block_args.input_filters == self._block_args.output_filters:
            self.residual = True

    self.skip_add = nn.quantized.FloatFunctional()
    self.sigmoid = nn.Sigmoid()

  def block_args(self):
    return self._block_args

  def _call_se(self, input_tensor):
    """Call Squeeze and Excitation layer.

    Args:
      input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.

    Returns:
      A output tensor, which should have the same shape as input.
    """
    se_tensor = torch.mean(input_tensor, self._spatial_dims, keepdim=True)
    se_tensor = self._se_expand(self._se_reduce(se_tensor))
    #print('Built Squeeze and Excitation with tensor shape: %s' %
    #                (se_tensor.shape))
    return self.sigmoid(se_tensor) * input_tensor

  def forward(self, inputs):
    """Implementation of MixnetBlock call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.

    Returns:
      A output tensor.
    """
    #tf.logging.info('Block input: %s shape: %s' % (inputs.name, inputs.shape))
    if self._block_args.expand_ratio != 1:
      x = self._expand_conv(inputs)
    else:
      x = inputs

    x = self._relu_fn(self._bn1(self._depthwise_conv(x)))

    if self._has_se:
        x = self._call_se(x)

    x = self._project_conv(x)
    if self.residual:
        x = self.skip_add.add(x, inputs)
    #tf.logging.info('Project: %s shape: %s' % (x.name, x.shape))
    return x


class MixnetModel(nn.Module):
  """A class implements tf.keras.Model for mixnet model.

    Reference: https://arxiv.org/abs/1807.11626
  """

  def __init__(self, blocks_args=None, global_params=None):
    """Initializes an `MixnetModel` instance.

    Args:
      blocks_args: A list of BlockArgs to construct Mixnet block modules.
      global_params: GlobalParams, a set of global parameters.

    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
    super(MixnetModel, self).__init__()
    if not isinstance(blocks_args, list):
      raise ValueError('blocks_args should be a list.')
    self._global_params = global_params
    self._blocks_args = blocks_args
    # Use relu in default for head and stem.
    self._relu_fn = nn.ReLU(inplace=True)
    self.endpoints = None

    """Builds a Mixnet model."""
    batch_norm_momentum = 1 - self._global_params.batch_norm_momentum
    epsilon = self._global_params.batch_norm_epsilon
    channel_axis = 1
    # Stem part.
    stem_size = self._global_params.stem_size
    self._conv_stem = ConvBNReLU(
        3,
        round_filters(stem_size, self._global_params),
        3, 2, padding=1,bias=False, momentum=batch_norm_momentum, eps=epsilon)

    self._blocks = nn.ModuleList()
    # Builds blocks.
    for block_args in self._blocks_args:
      assert block_args.num_repeat > 0
      # Update block input and output filters based on depth multiplier.
      block_args = block_args._replace(
          input_filters=round_filters(block_args.input_filters,
                                      self._global_params),
          output_filters=round_filters(block_args.output_filters,
                                       self._global_params))

      # The first block needs to take care of stride and filter size increase.
      self._blocks.append(MixnetBlock(block_args, self._global_params))
      if block_args.num_repeat > 1:
        # pylint: disable=protected-access
        block_args = block_args._replace(
            input_filters=block_args.output_filters, strides=[1, 1])
        # pylint: enable=protected-access
      for _ in xrange(block_args.num_repeat - 1):
        self._blocks.append(MixnetBlock(block_args, self._global_params))


    # Head part.
    self._conv_head = ConvBNReLU(
        block_args.output_filters,
        self._global_params.feature_size,
        1, 1, padding=0, bias=False, momentum=batch_norm_momentum, eps=epsilon)


  def forward(self, inputs):
    """Implementation of MixnetModel call().

    Args:
      inputs: input tensors.
      training: boolean, whether the model is constructed for training.
      features_only: build the base feature network only.

    Returns:
      output tensors.
    """
    outputs = None
    # Calls Stem layers
    outputs = self._conv_stem(inputs)
    # Calls blocks.
    reduction_idx = 0
    for idx, block in enumerate(self._blocks):
        outputs = block(outputs)

    outputs = self._conv_head(outputs)
    return outputs



class MixnetDecoder(object):
  """A class of Mixnet decoder to get model configuration."""

  def _decode_block_string(self, block_string):
    """Gets a mixnet block through a string notation of arguments.

    E.g. r2_k3_a1_p1_s2_e1_i32_o16_se0.25_noskip: r - number of repeat blocks,
    k - kernel size, s - strides (1-9), e - expansion ratio, i - input filters,
    o - output filters, se - squeeze/excitation ratio

    Args:
      block_string: a string, a string representation of block arguments.

    Returns:
      A BlockArgs instance.
    Raises:
      ValueError: if the strides option is not correctly specified.
    """
    assert isinstance(block_string, str)
    ops = block_string.split('_')
    options = {}
    for op in ops:
      splits = re.split(r'(\d.*)', op)
      if len(splits) >= 2:
        key, value = splits[:2]
        options[key] = value

    if 's' not in options or len(options['s']) != 2:
      raise ValueError('Strides options should be a pair of integers.')

    def _parse_ksize(ss):
      return [int(k) for k in ss.split('.')]

    return BlockArgs(
        expand_ksize=_parse_ksize(options['a']),
        dw_ksize=_parse_ksize(options['k']),
        project_ksize=_parse_ksize(options['p']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        id_skip=('noskip' not in block_string),
        se_ratio=float(options['se']) if 'se' in options else None,
        strides=[int(options['s'][0]), int(options['s'][1])],
        swish=('sw' in block_string),
        dilated=('dilated' in block_string))

  def _encode_block_string(self, block):
    """Encodes a Mixnet block to a string."""
    def _encode_ksize(arr):
      return '.'.join([str(k) for k in arr])

    args = [
        'r%d' % block.num_repeat,
        'k%s' % _encode_ksize(block.dw_ksize),
        'a%s' % _encode_ksize(block.expand_ksize),
        'p%s' % _encode_ksize(block.project_ksize),
        's%d%d' % (block.strides[0], block.strides[1]),
        'e%s' % block.expand_ratio,
        'i%d' % block.input_filters,
        'o%d' % block.output_filters
    ]
    if (block.se_ratio is not None and block.se_ratio > 0 and
        block.se_ratio <= 1):
      args.append('se%s' % block.se_ratio)
    if block.id_skip is False:  # pylint: disable=g-bool-id-comparison
      args.append('noskip')
    if block.swish:
      args.append('sw')
    if block.dilated:
      args.append('dilated')
    return '_'.join(args)

  def decode(self, string_list):
    """Decodes a list of string notations to specify blocks inside the network.

    Args:
      string_list: a list of strings, each string is a notation of Mixnet
        block.build_model_base

    Returns:
      A list of namedtuples to represent Mixnet blocks arguments.
    """
    assert isinstance(string_list, list)
    blocks_args = []
    for block_string in string_list:
      blocks_args.append(self._decode_block_string(block_string))
    return blocks_args

  def encode(self, blocks_args):
    """Encodes a list of Mixnet Blocks to a list of strings.

    Args:
      blocks_args: A list of namedtuples to represent Mixnet blocks arguments.
    Returns:
      a list of strings, each string is a notation of Mixnet block.
    """
    block_strings = []
    for block in blocks_args:
      block_strings.append(self._encode_block_string(block))
    return block_strings


def mixnet_s(depth_multiplier=None):
  """Creates mixnet-s model.

  Args:
    depth_multiplier: multiplier to number of filters per layer.

  Returns:
    blocks_args: a list of BlocksArgs for internal Mixnet blocks.
    global_params: GlobalParams, global parameters for the model.
  """
  blocks_args = [
      'r1_k3_a1_p1_s11_e1_i16_o16',
      'r1_k3_a1.1_p1.1_s22_e6_i16_o24',
      'r1_k3_a1.1_p1.1_s11_e3_i24_o24',

      'r1_k3.5.7_a1_p1_s22_e6_i24_o40_se0.5_sw',
      'r3_k3.5_a1.1_p1.1_s11_e6_i40_o40_se0.5_sw',

      'r1_k3.5.7_a1_p1.1_s22_e6_i40_o80_se0.25_sw',
      'r2_k3.5_a1_p1.1_s11_e6_i80_o80_se0.25_sw',

      'r1_k3.5.7_a1.1_p1.1_s11_e6_i80_o120_se0.5_sw',
      'r2_k3.5.7.9_a1.1_p1.1_s11_e3_i120_o120_se0.5_sw',

      'r1_k3.5.7.9.11_a1_p1_s22_e6_i120_o200_se0.5_sw',
      'r2_k3.5.7.9_a1_p1.1_s11_e6_i200_o200_se0.5_sw',
  ]
  global_params = GlobalParams(
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=0.2,
      data_format='channels_last',
      num_classes=1000,
      depth_multiplier=depth_multiplier,
      depth_divisor=8,
      min_depth=None,
      stem_size=16,
      use_keras=True,
      feature_size=320)
  decoder = MixnetDecoder()
  return decoder.decode(blocks_args), global_params


def mixnet_m(depth_multiplier=None):
  """Creates a mixnet-m model.

  Args:
    depth_multiplier: multiplier to number of filters per layer.

  Returns:
    blocks_args: a list of BlocksArgs for internal Mixnet blocks.
    global_params: GlobalParams, global parameters for the model.
  """
  blocks_args = [
      'r1_k3_a1_p1_s11_e1_i24_o24',
      'r1_k3.5.7_a1.1_p1.1_s22_e6_i24_o32',
      'r1_k3_a1.1_p1.1_s11_e3_i32_o32',

      'r1_k3.5.7.9_a1_p1_s22_e6_i32_o40_se0.5_sw',
      'r3_k3.5_a1.1_p1.1_s11_e6_i40_o40_se0.5_sw',

      'r1_k3.5.7_a1_p1_s22_e6_i40_o80_se0.25_sw',
      'r3_k3.5.7.9_a1.1_p1.1_s11_e6_i80_o80_se0.25_sw',

      'r1_k3_a1_p1_s11_e6_i80_o120_se0.5_sw',
      'r3_k3.5.7.9_a1.1_p1.1_s11_e3_i120_o120_se0.5_sw',

      'r1_k3.5.7.9_a1_p1_s22_e6_i120_o200_se0.5_sw',
      'r3_k3.5.7.9_a1_p1.1_s11_e6_i200_o200_se0.5_sw',
  ]
  global_params = GlobalParams(
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=0.25,
      data_format='channels_last',
      num_classes=1000,
      depth_multiplier=depth_multiplier,
      depth_divisor=8,
      min_depth=None,
      stem_size=24,
      use_keras=True,
      feature_size=320)
  decoder = MixnetDecoder()
  return decoder.decode(blocks_args), global_params


def mixnet_l(depth_multiplier=None):
  d = 1.3 * depth_multiplier if depth_multiplier else 1.3
  return mixnet_m(d)


def get_model_params(model_name, override_params):
  """Get the block args and global params for a given model."""
  if model_name == 'mixnet-s':
    blocks_args, global_params = mixnet_s()
  elif model_name == 'mixnet-m':
    blocks_args, global_params = mixnet_m()
  elif model_name == 'mixnet-l':
    blocks_args, global_params = mixnet_l()
  else:
    raise NotImplementedError('model name is not pre-defined: %s' % model_name)

  if override_params:
    # ValueError will be raised here if override_params has fields not included
    # in global_params.
    global_params = global_params._replace(**override_params)
  return blocks_args, global_params


def build_model(model_name):
  """A helper functiion to create a Mixnet model and return predicted logits.

  Args:
    images: input images tensor.
    model_name: string, the model name of a pre-defined Mixnet.
    training: boolean, whether the model is constructed for training.
    override_params: A dictionary of params for overriding. Fields must exist in
      mixnet_model.GlobalParams.

  Returns:
    logits: the logits tensor of classes.
    endpoints: the endpoints for each layer.
  Raises:
    When model_name specified an undefined model, raises NotImplementedError.
    When override_params has invalid fields, raises ValueError.
  """
  blocks_args, global_params = get_model_params(model_name, None)
  print('blocks_args= {}'.format(blocks_args))
  print('global_params= {}'.format(global_params))
  model = MixnetModel(blocks_args, global_params)

  return model


if __name__ == '__main__':
    model = build_model(model_name='mixnet-l')

    input = torch.ones((1,3,256,256))
    output = model(input)
    pass







