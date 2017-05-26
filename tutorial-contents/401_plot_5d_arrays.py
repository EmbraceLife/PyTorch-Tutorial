import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import numpy as np

#### plot (4,4,5,5)
data_4d = np.linspace(0, 1, 400).reshape((4,4,5,5))
num_out_img, num_inner_img, img_w, img_h = data_4d.shape

fig = plt.figure(1, figsize=(6, 6))

outer_grid = math.ceil(math.sqrt(num_out_img))
inner_grid = math.ceil(math.sqrt(num_inner_img))

outer_frame = gridspec.GridSpec(outer_grid, outer_grid)

for sub in range(num_out_img):

	inner_frame = gridspec.GridSpecFromSubplotSpec(inner_grid, inner_grid, subplot_spec=outer_frame[sub], wspace=0.0, hspace=0.0)

	for sub_sub in range(num_inner_img):

		ax = plt.Subplot(fig, inner_frame[sub_sub])
		ax.imshow(data_4d[sub, sub_sub, :, :], cmap='gray')
		fig.add_subplot(ax)

plt.show()

#### plot (4,4,4,5,5)
data_5d = np.linspace(0, 1, 1600).reshape((4,4,4,5,5))
num_out_img, num_inner_img, num_deep_img, img_w, img_h = data_5d.shape

fig = plt.figure(1, figsize=(6, 6))

outer_grid = math.ceil(math.sqrt(num_out_img))
inner_grid = math.ceil(math.sqrt(num_inner_img))
deep_grid = math.ceil(math.sqrt(num_deep_img))

outer_frame = gridspec.GridSpec(outer_grid, outer_grid)

for sub in range(num_out_img):

	inner_frame = gridspec.GridSpecFromSubplotSpec(inner_grid, inner_grid, subplot_spec=outer_frame[sub], wspace=0.0, hspace=0.0)

	for sub_sub in range(num_inner_img):

		deep_frame = gridspec.GridSpecFromSubplotSpec(deep_grid, deep_grid, subplot_spec=inner_frame[sub_sub], wspace=0.0, hspace=0.0)

		for deep in range(num_deep_img):

			ax = plt.Subplot(fig, deep_frame[deep])
			ax.imshow(data_5d[sub, sub_sub, deep, :, :], cmap='gray')
			fig.add_subplot(ax)

plt.show()
