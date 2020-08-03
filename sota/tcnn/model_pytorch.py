import torch
import torch.nn as nn


class conv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride,  padding, use_bn = True):
		super().__init__()

		if(use_bn):
			self.cnn = nn.Sequential(
				nn.Conv2d(
					in_channels = in_channels,
					out_channels = out_channels,                       
					kernel_size = (kernel_size, 1),                # e.g. (9,  1)
					stride = (stride, 1),                         # e.g. (1, 1)
					padding = (padding, 0),                            # e.g. (4, 0)
				),
				nn.BatchNorm2d(out_channels),
				nn.ReLU(),
			)
		else:
			self.cnn = nn.Sequential(
				nn.Conv2d(
					in_channels = in_channels,
					out_channels = out_channels,                       
					kernel_size = (kernel_size, 1),                # e.g. (9,  1)
					stride = (stride, 1),                         # e.g. (1, 1)
					padding = (padding, 0),                            # e.g. (4, 0)
				),
			)



	def __call__(self, x):
		# x shape must be (batch_size, in_channels, height, width)
		# or (batch_size, in_channels, obs_len, 1)  for 1D convolution over observed traj

		return self.cnn(x)
		
class deconv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride,  padding, dropout = 0.2):
		super().__init__()


		self.cnn = nn.Sequential(
			nn.ConvTranspose2d(
				in_channels = in_channels,
				out_channels = out_channels,                       
				kernel_size = (kernel_size, 1),                # e.g. (9,  1)
				stride = (stride, 1),                         # e.g. (1, 1)
				padding = (padding, 0),                            # e.g. (4, 0)
			),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
		)

	def __call__(self, x):
		# x shape must be (batch_size, in_channels, height, width)
		# or (batch_size, in_channels, obs_len, 1)  for 1D convolution over observed traj

		return self.cnn(x)
		


class TCNN(nn.Module):
	def __init__(self, pred_len = 10):
		super().__init__()

		self.pred_len = pred_len
		self.encoder = nn.Sequential(
			conv2d(in_channels = 2, out_channels = 32, kernel_size = 3, stride = 1,  padding = 0), 
			conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1,  padding = 0), 
			conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1,  padding = 0),
			conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1,  padding = 0) 
		)

		self.intermediate = conv2d(in_channels = 128, out_channels = 128, kernel_size =1, stride = 1,  padding = 0)

		self.decoder = nn.Sequential(
			conv2d(in_channels = 128, out_channels = 256, kernel_size = 1, stride = 1,  padding = 0), 
			conv2d(in_channels = 256, out_channels = 256, kernel_size = 1, stride = 1,  padding = 0), 
			deconv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1,  padding = 0), 
			deconv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1,  padding = 0), 
			deconv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1,  padding = 0), 
			deconv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1,  padding = 0), 
			conv2d(in_channels = 32, out_channels = 2, kernel_size = 1, stride = 1,  padding = 0, use_bn = False)
		)

		# n_extra_deconv2d = (pred_len - 10)/2 
		# for i in range(n_extra_deconv2d):
		# 	self.decoder.add_module(deconv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1,  padding = 0))




	def __call__(self, x):	

		x = x.permute(0, 2, 1).contiguous()
		x = x.unsqueeze(3)					# x ~ (batch_size, in_channels (2), obs_len (10), 1)

		y = self.encoder(x)
		y = self.intermediate(y)
		y = self.decoder(y)					# y ~ (batch_size, out_channels, pred_len (10), 1)
		y = self.last(y)

		y = y.squeeze(3)					# x ~ (batch_size, in_channels (2), obs_len (10), 1)
		y = y.permute(0, 2, 1).contiguous()

		return y


class TCNN_POSE(nn.Module):
	def __init__(self, pred_len = 10):
		super().__init__()

		self.pred_len = pred_len

		self.encoder_location = nn.Sequential(
			conv2d(in_channels = 2, out_channels = 32, kernel_size = 3, stride = 1,  padding = 0), 
			conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1,  padding = 0), 
			conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1,  padding = 0),
			conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1,  padding = 0) 
		)

		self.encoder_pose = nn.Sequential(
			conv2d(in_channels = 75, out_channels = 32, kernel_size = 3, stride = 1,  padding = 0), 
			conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1,  padding = 0), 
			conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1,  padding = 0),
			conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1,  padding = 0) 
		)


		self.intermediate = conv2d(in_channels = 128*2, out_channels = 128, kernel_size =1, stride = 1,  padding = 0)

		self.decoder = nn.Sequential(
			conv2d(in_channels = 128, out_channels = 256, kernel_size = 1, stride = 1,  padding = 0), 
			conv2d(in_channels = 256, out_channels = 256, kernel_size = 1, stride = 1,  padding = 0), 
			deconv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1,  padding = 0), 
			deconv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1,  padding = 0), 
			deconv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1,  padding = 0), 
			deconv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1,  padding = 0), 
			conv2d(in_channels = 32, out_channels = 2, kernel_size = 1, stride = 1,  padding = 0, use_bn = False), 
		)

		#self.last = nn.Linear(2, 2)

	def __call__(self, x, pose ):	


		x = x.permute(0, 2, 1).contiguous()
		x = x.unsqueeze(3)					# x ~ (batch_size, in_channels (2), obs_len (10), 1)


		pose = pose.permute(0, 2, 1).contiguous()
		pose = pose.unsqueeze(3)					# x ~ (batch_size, in_channels (25)), obs_len (10), 1)


		x = self.encoder_location(x)
		pose_y = self.encoder_pose(pose)

		#print(x.shape)

		#print(pose_y.shape)

		encoded_f = torch.cat((x, pose_y), dim = 1)
		#print(encoded_f.shape)
		#input("here")


		y = self.intermediate(encoded_f)
		y = self.decoder(y)					# y ~ (batch_size, out_channels, pred_len (10), 1)
		#y = self.last(y)

		y = y.squeeze(3)					# x ~ (batch_size, in_channels (2), obs_len (10), 1)
		y = y.permute(0, 2, 1).contiguous()

		return y