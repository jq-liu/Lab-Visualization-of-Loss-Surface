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
import torch
import copy



surface_path = './loss_surface/'
model_path = './checkpoint/CNN_AVE-8'
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
# con.load_test_data()
con.set_plt_model(model[args.model_name])
x = 150
y = 15
num_axis = 20
xcoordinates = np.linspace(-x, x, num_axis)
ycoordinates = np.linspace(-y, y, num_axis)
coordinates = [xcoordinates, ycoordinates]

model_files = []
for epoch in range(13):
	model_file = './checkpoint/CNN_AVE-' + str(epoch)
	model_files.append(model_file)
con.pltModel.load_state_dict(torch.load(model_files[-1]))
w = copy.deepcopy(con.get_weights())
# d = copy.deepcopy(con.get_direction())
d = con.setup_PCA_directions(model_files, w)

train_losses = np.loadtxt(surface_path + 'train_losses')
# train_losses = con.crunch(coordinates, model_path, w, d)
# np.savetxt(surface_path + "train_losses", train_losses)

# con.plot_2d_contour(surface_path+"2D_Contour", coordinates, train_losses, vmin=0.01, vmax=500, vlevel=2)
print("PLT CONTOUR END!")
proj_file = con.project_trajectory(d, w, model_files)
con.plot_contour_trajectory(train_losses, d, proj_file, coordinates, surface_path, vmin=0.0001, vmax=10, vlevel=0.2)

