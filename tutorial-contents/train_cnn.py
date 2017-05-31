
def train(args):
	""" Trains a model.
	"""
	# prepare dataset
	train_loader, test_images, test_labels = prepareData(args)

	# build net
	cnn, optimizer, loss_func, cnn2pp = build_net(args)

	# storage boxes
	losses = []
	steps = []

	# plotting while training or not
	if args.display:
		plt.ion()

	# for every epoch of training
	for epoch_idx in range(args.num_epochs):

		# loss value has to be carried in and out
		loss = None

		# traing model for every batch
		for batch_idx, (batch_img, batch_lab) in enumerate(train_loader):

			b_img = Variable(batch_img)
			b_lab = Variable(batch_lab)

			conv1_relu, conv1_maxpool, conv2_relu, conv2_maxpool, logits = cnn(b_img)
			loss = loss_func(logits, b_lab)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print("loss for batch_{}: %.4f".format(batch_idx) % loss.data.numpy()[0])
			# don't train the full epoch or total_num_batches, but only specific num_batches in each epoch
			if args.num_batches == batch_idx:
				break


		# plot every n epochs
		if args.plotting == True and epoch_idx % 1 == 0:

			# keep test and plot based on a single and same image
			conv1_relu, conv1_maxpool, conv2_relu, conv2_maxpool, logits = cnn(torch.unsqueeze(test_images[0], dim=0))
			# Note: input_image has to be 4d tensor (n, 1, 28, 28)

			# every time when plotting, update losses and steps
			losses.append(loss.data.numpy().tolist()[0])
			steps.append(epoch_idx+1)

			# every time when plotting, update values of x, y, weights, biases, activations, loss
			param_names = []
			param_values = []
			for k, v in cnn.state_dict().items():
			    param_names.append(k)
			    param_values.append(v)

			## insert conv2_maxpool and conv2_relu
			param_names.insert(4, "2maxPool")
			param_names.insert(4, "2relu")
			param_values.insert(4, conv2_maxpool.data[0])
			param_values.insert(4, conv2_relu.data[0])

			## insert conv1_maxpool and conv1_relu
			param_names.insert(2, "1maxPool")
			param_names.insert(2, "1relu")
			param_values.insert(2, conv1_maxpool.data[0])
			param_values.insert(2, conv1_relu.data[0])

			## insert a single image and its label
			param_names.insert(0, "image")
			test_img1 = test_images.data.numpy()[0] # (1, 28, 28)
			np_img1 = np.squeeze(test_img1) # (28, 28)
			test_lab1 = test_labels[0]
			# insert a single image and label for plotting loop
			param_values.insert(0, (np_img1, test_lab1))


			## append logits for a single images
			logits1 = logits[0]
			logits1_softmax = F.softmax(logits1).data
			param_names.append("softmax")
			param_values.append(logits1_softmax)

			## append losses and steps
			# losses.append(loss.data[0])
			# steps.append(t)
			param_names.append("loss")
			param_values.append([steps, losses])
			# check size of all layers except image and loss
			# pp [p.size() for p in param_values[1:-1]]

			# shorten param_names
			shorten_names = [p_name.replace("weight", "w").replace("bias", "b") for p_name in param_names]
			param_names = shorten_names

			if args.display:
				display(args, param_names, param_values, cnn)

			else:
				saveplots(args, param_names, param_values, cnn)

			# epoch counting (start 1 not 0)
			print("finish saving plot for epoch_%d" % epoch_idx+1)

	if args.display:
		plt.ioff()
	else:
		# save net and log
		torch.save(cnn, args.net_path)
		torch.save((steps, losses), args.log_path)
		# convert saved images to gif (speed up, down, normal versions)
		# img2gif(args)
