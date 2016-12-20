## plotting based on an image number
def plot_image_num(image_num):
	from cifar_load import unpickle, load_cifar
	import matplotlib.pyplot as plt
	from PIL import Image

	# loading data and definitions
	data = load_cifar()
	images = data[0]
	images_index = data[1]
	images_labels = data[2]

	# returning image label

	# plotting the image
	images = images.reshape(50000,32,32, 3)#.transpose(0,2,3,1)
	images = images.reshape(50000,3,32,32).transpose(0,2,3,1)
	plt.imshow(images[:][image_num] , interpolation='spline16')
	plt.show()

	label = images_labels[images_index[image_num]]
	return label

## plotting based on keyword (must be in fine labels)
# image_num defines which one of the 500 images for the specific keyword will be plotted
def plot_label(keyword, image_num):
	from cifar_load import unpickle, load_cifar
	import matplotlib.pyplot as plt
	from PIL import Image

	# loading data and definitions
	data = load_cifar()
	images = data[0]
	images_index = data[1]
	images_labels = data[2]

	# find all image numbers
	keyword_index = images_labels.index(keyword)
	image_numbers = [i for i,x in enumerate(images_index) if x == keyword_index]

	plot_image_num(image_numbers[image_num])











