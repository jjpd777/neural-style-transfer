# USAGE
# python style_transfer.py

# import the necessary packages
from pyimagesearch.nn.conv import NeuralStyle
from keras.applications import VGG19
import os

styles = ["circuit-1.jpg","circuit-2.jpg","circuit-3.jpg","circuit-4.jpg",
		  "circuit-5.jpg","circuit-6.jpg","circuit-7.jpg"]
# initialize the settings dictionary

def create_styles(style_path,dst):
	SETTINGS = {
		# initialize the path to the input (i.e., content) image,
		# style image, and path to the output directory
		"input_path": "guacamaya-content.jpg",
		"style_path": style_path,
		"output_path": dst,

		# define the CNN to be used style transfer, along with the
		# set of content layer and style layers, respectively
		"net": VGG19,
		"content_layer": "block4_conv2",
		"style_layers": ["block1_conv1", "block2_conv1",
			"block3_conv1", "block4_conv1", "block5_conv1"],

		# store the content, style, and total variation weights,
		# respectively
		"content_weight": 1.0,
		"style_weight": 100.0,
		"tv_weight": 7.5,

		# number of iterations
		"iterations": 75,
	}
	return SETTINGS

for style in styles:
	style_path = "circuits/"+style
	name = style.split(".")[0]
	output_dir = "output/"+name
	os.makedir(dir_name)
	settings = create_styles(style_path,output_dir)
	ns = NeuralStyle(settings)
	ns.transfer()
