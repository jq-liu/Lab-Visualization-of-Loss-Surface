import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
import copy
import torch


surface_path = './loss_surface/'
mdoel_path = './checkpoint/CNN_AVE-18'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'cnn_ave', help = 'name of the model')
args = parser.parse_args()
model = {
	'pcnn_att': models.PCNN_ATT,
	'pcnn_one': models.PCNN_ONE,
	'pcnn_ave': models.PCNN_AVE,
	'cnn_att': models.CNN_ATT,
	'cnn_one': models.CNN_ONE,
	'cnn_ave': models.CNN_AVE
}

con = config.Config()
con.load_train_data()
con.set_plt_model(model[args.model_name])
x = 1
y = 1
num_axis = 20
xcoordinates = np.linspace(-x, x, num_axis)
ycoordinates = np.linspace(-y, y, num_axis)
coordinates = [xcoordinates, ycoordinates]

con.pltModel.load_state_dict(torch.load(mdoel_path))
w = copy.deepcopy(con.get_weights())
d = copy.deepcopy(con.get_direction())

# train_losses = con.crunch(coordinates, mdoel_path, w, d)
# np.savetxt(surface_path + "train_losses_pre_0", train_losses)
print("PLT END1!")
train_losses = np.loadtxt(surface_path + 'train_losses_pre_0')
con.plot_2d_contour(surface_path+"2Dcontour_pre_0", coordinates, train_losses, vmin=0.01, vmax=1, vlevel=0.02)



