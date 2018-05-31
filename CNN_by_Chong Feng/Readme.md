CNN Mission Recognition by Chong Feng

python code:

	get_images.py    : Capture images from the game. Change the variable MISSION to change the game map.

	display_layers.py: Output the feature maps after each sublayer of the first CNN layer. 
	                   Opening sublayers used and closing sublayers not used manually are required.

	augmentation.py  : Do the data augmentation. 
	                   Uncomment one operation each time and run it for several times.

	split.py         : Split the images to training and validation data sets.

	CNN.py           : Train the CNN classifier. Changing the number of layers manually is required. 
	                   Validation is called test in the code.

	test.py          : Test the CNN classifiers with the original images captured from the game.

Directory:

	Application:

		application.py: A simple application using the CNN classifier to do the mission recognition. 
	                        Change the variable MISSION to change the game map.

		ckpts         : The saved CNN classifier.

		images        : The images captured for vote for the mission.

	data:
	
		augmentation:

			original    : The original images used for data augmentation.
			output      : The output images by data augmentation.
			split       : The splited training and validation images and their labels. 
		                      Validation is called test here. 

		augmentation_example: Some data augmentation examples used for report.

		layers              : Some feature maps for report after each sublayer of the first CNN layer.

		original            : The original iamges captured from the game and their labels used for test.

		preprocessed        : Preprocessed images of the original 200 images which are not used 
	                              in the code but mentioned in the report.
