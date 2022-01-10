
__author__ = 'Majd Jamal'

import argparse

parser = argparse.ArgumentParser(description='Image Classification of fruits and vegetables')

parser.add_argument('--train',  action = 'store_true', default=False,
	help='Run the training script.')

parser.add_argument('--predict',  action = 'store_true', default=False,
	help='Run the training script.')


args = parser.parse_args()
