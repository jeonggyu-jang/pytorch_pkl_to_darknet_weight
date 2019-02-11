# All conv layers or fc layers must have the same bn condition (conv and fc not have to be same condition)
# ex1) conv1 bn1 conv2 bn2 fc1 fc2 => Good
# ex2) conv1 conv2 bn2 conv3 fc1 bn1 fc2 => Not Good..

import torch
import numpy
model_name = 'ConvNet_pruned'
model = torch.load(model_name+'.pkl',map_location='cpu')

bn_layer_name_tags = ['bn','BN','batchnorm','Batchnorm','BatchNorm','batchnormalization','Batchnormalization','BatchNormalization','BATCHNORMALIZATION']
conv_layer_name_tags = ['conv','Conv','convolution','Convolution','CONV']
fc_layer_name_tags = ['fc','FC','fullyconnected','Fullyconnected','FullyConnected','FULLYCONNECTED']

def matching_name_tag(tag):
	for bn_tag in bn_layer_name_tags:
		if bn_tag in tag:
			if 'weight' in tag:
				return 'bn_weight'
			elif 'bias' in tag:
				return 'bn_bias'
			elif 'running_mean' in tag:
				return 'bn_running_mean'
			elif 'running_var' in tag:
				return 'bn_running_var'
	for conv_tag in conv_layer_name_tags:
		if conv_tag in tag:
			if 'weight' in tag:
				return 'conv_weight'
			elif 'bias' in tag:
				return 'conv_bias'
	for fc_tag in fc_layer_name_tags:
		if fc_tag in tag:
			if 'weight' in tag:
				return 'fc_weight'
			elif 'bias' in tag:
				return 'fc_bias'

output_list = []
conv_layer_flag = 0
fc_layer_flag = 0
bn_layer_flag = 0

for i,m in enumerate(model):
	tag = matching_name_tag(m)
	print('------ tag : '+tag)
	if conv_layer_flag > 2 or fc_layer_flag > 2:
		print('[ERROR] conv or fc layer is overloaded!!\n')
	if ('conv' in tag) or ('fc' in tag):
		if conv_layer_flag == 2: 
			output_list += conv_bias
			print('conv_bias\n')
			if bn_layer_flag == 4:
				output_list += bn_weight
				output_list += bn_running_mean
				output_list += bn_running_var
				print('conv_bn\n')
				bn_layer_flag = 0
			output_list += conv_weight
			print('conv_weight\n')
			conv_layer_flag = 0
		elif fc_layer_flag == 2:
			output_list += fc_bias
			print('fc_bias\n')
			output_list += fc_weight
			print('fc_weight\n')
			if bn_layer_flag == 4:
				output_list += bn_weight
				output_list += bn_running_mean
				output_list += bn_running_var
				print('fc_bn\n')
				bn_layer_flag = 0
			fc_layer_flag = 0
	
		if tag == 'conv_weight':
			conv_layer_flag += 1
			conv_weight = (list(model[m].reshape(-1).numpy()))
		elif tag == 'conv_bias':
			conv_layer_flag += 1
			conv_bias = (list(model[m].reshape(-1).numpy()))
		elif tag == 'fc_weight':
			fc_layer_flag += 1
			fc_weight = (list(model[m].reshape(-1).numpy()))
		elif tag == 'fc_bias':
			fc_layer_flag += 1
			fc_bias = (list(model[m].reshape(-1).numpy()))		

	elif 'bn' in tag: 
		if bn_layer_flag == 4:
			print('[ERROR] duplicated bn!!\n')
		if tag == 'bn_weight':
			bn_layer_flag += 1
			bn_weight = (list(model[m].reshape(-1).numpy()))
		elif tag == 'bn_bias':
			bn_layer_flag += 1
			bn_bias = (list(model[m].reshape(-1).numpy()))
		elif tag == 'bn_running_mean':
			bn_layer_flag += 1
			bn_running_mean = (list(model[m].reshape(-1).numpy()))
		elif tag == 'bn_running_var':
			bn_layer_flag += 1
			bn_running_var = (list(model[m].reshape(-1).numpy()))

#for last layer 
if conv_layer_flag == 2: 
	output_list += conv_bias
	print('conv_bias\n')
	if bn_layer_flag == 4:
		output_list += bn_weight
		output_list += bn_running_mean
		output_list += bn_running_var
		print('conv_bn\n')
		bn_layer_flag = 0
	output_list += conv_weight
	print('conv_weight\n')
	conv_layer_flag = 0
elif fc_layer_flag == 2:
	output_list += fc_bias
	print('fc_bias\n')
	output_list += fc_weight
	print('fc_weight\n')
	if bn_layer_flag == 4:
		output_list += bn_weight
		output_list += bn_running_mean
		output_list += bn_running_var
		print('fc_bn\n')
		bn_layer_flag = 0
	fc_layer_flag = 0

numpy.savetxt(model_name+'.csv',output_list,delimiter='\n',fmt='%f')

