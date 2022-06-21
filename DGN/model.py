import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
class Encoder(nn.Module):
	def __init__(self, din=22, hidden_dim=128):
		super(Encoder, self).__init__()
		self.fc = nn.Linear(din, 128)
		self.fc1 = nn.Linear(128, hidden_dim)
	#@autocast
	def forward(self, x):
		embedding = F.relu(self.fc(x))
		embedding = F.relu(self.fc1(embedding))
		return embedding

class AttModel(nn.Module):
	def __init__(self, n_node, din, hidden_dim, dout):
		super(AttModel, self).__init__()
		self.fcv = nn.Linear(din, hidden_dim)
		self.fck = nn.Linear(din, hidden_dim)
		self.fcq = nn.Linear(din, hidden_dim)
		self.fcout = nn.Linear(hidden_dim, dout)
	#@autocast
	def forward(self, x, mask):
		v = F.relu(self.fcv(x))
		q = F.relu(self.fcq(x))
		k = F.relu(self.fck(x)).permute(0,2,1)
		att = F.softmax(torch.mul(torch.bmm(q,k), mask) - 9e15*(1 - mask),dim=2)

		out = torch.bmm(att,v)
		# out = torch.bmm(mask,v) #commnet
		# out = torch.add(out,v)
		out = F.relu(self.fcout(out))
		return out

class Q_Net(nn.Module):
	def __init__(self, hidden_dim, dout):
		super(Q_Net, self).__init__()
		self.fc = nn.Linear(hidden_dim, dout)

	def forward(self, x):
		q = self.fc(x)
		return q

class DGN(nn.Module):
	def __init__(self,n_agent,num_inputs,hidden_dim,num_actions):
		super(DGN, self).__init__()
		
		self.encoder = Encoder(num_inputs,hidden_dim)
		self.att_1 = AttModel(n_agent,hidden_dim,hidden_dim,hidden_dim)
		self.att_2 = AttModel(n_agent,hidden_dim,hidden_dim,hidden_dim)
		self.q_net = Q_Net(hidden_dim,num_actions)
	
	#@autocast
	def forward(self, x, mask):
		h1 = self.encoder(x)
		h2 = self.att_1(h1, mask)
		h3 = self.att_2(h2, mask)
		#h4 = self.att_2(h3, mask)

		q = self.q_net(h3)
		return q 
