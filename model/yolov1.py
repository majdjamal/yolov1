
__author__ = 'Majd Jamal'
#Inspiration from https://youtu.be/n9_XyCGr-MI

import numpy as np
import tensorflow as tf
from model._network_config import architecture
from tensorflow import random_normal_initializer
from tensorflow.keras import layers, losses, Sequential, Model

class YoloV1:

	def __init__(self):

		self.S = 7    #Grid size
		self.B = 2    #Number of boxes for each cell
		self.C = 2    #Number of classes


	def ConvBlock(self, filters: int, kernel_size: int, stride: int, apply_batchnorm: bool = True) -> Sequential:
		""" Convolutional block. Used to build YOLOv1 network arcitecture.
		:params filters: number of output channels
		:params kernel_size: kernel kernel_size
		:params apply_batchnorm: To apply batch normalization
		:return block: A downscaling block.
		"""
		initializer = random_normal_initializer(0., 0.02)

		block = Sequential()

		block.add(
			layers.Conv2D(filters, kernel_size,
			strides = stride,
			padding = "same",
			kernel_initializer = initializer,
			use_bias = False)
		)

		if apply_batchnorm:

			block.add(layers.BatchNormalization())

		block.add(layers.LeakyReLU(0.1))

		return block

	def FinalBlock(self, S: int, B: int, C: int) -> Sequential:
		"""
        This function implement the YoloV1 bottleneck.
		:params S: Grid Size,
		:params B: Number of Boxes on each cell
		:params C: Number of Classes
        :return block: bottleneck
		"""

		block = Sequential()

		block.add(layers.Flatten())
		block.add(layers.Dense(496))
		block.add(layers.Dropout(0.0))
		block.add(layers.LeakyReLU(0.1))
		block.add(layers.Dense(S*S* (C + B * 5)))
		block.add(layers.Reshape((S,S, C + B * 5)))

		return block

	def MakeYOLO(self) -> Model:
		""" Builds the detector model.

		:return: Non-compiled detector
		"""

		input = layers.Input((448,448,3))

		stack = []

		for block_id in architecture:

			if isinstance(block_id, tuple):
				kernel_size, output, stride, padding = block_id

				block = self.ConvBlock(
						filters = output ,
						kernel_size = (kernel_size, kernel_size),
						stride = (stride, stride)
						)

				stack.append(block)

			elif isinstance(block_id, str):

				block = layers.MaxPool2D(
					pool_size = (2,2),
					strides = (2,2))

				stack.append(block)

			elif isinstance(block_id, list):

				kernel_size_1, output_1, stride_1, padding_1 = block_id[0]

				kernel_size_2, output_2, stride_2, padding_2 = block_id[1]

				repeat_int = block_id[2]

				block_1 = self.ConvBlock(
						filters = output_1 ,
						kernel_size = (kernel_size_1, kernel_size_1),
						stride = (stride_1, stride_1)
						)

				block_2 = self.ConvBlock(
						filters = output_2 ,
						kernel_size = (kernel_size_2, kernel_size_2),
						stride = (stride_2, stride_2)
						)

				for _ in range(repeat_int):
					stack.append(block_1)
					stack.append(block_2)

		final_block = self.FinalBlock(self.S, self.B, self.C)

		x = input

		for block in stack:
			x = block(x)

		x = final_block(x)

		return Model(inputs = input, outputs = x)
