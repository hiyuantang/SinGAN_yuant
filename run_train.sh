# Make sure cd into the src directory to run this bash shell

# Models trained on balloons
python train.py --input_image balloons_downsampled.png # model 1
python train_patches.py --input_image balloons_downsampled.png # model 2

python train.py --input_image balloons_downsampled.png --number_of_noises 12 # model 3
python train.py --input_image balloons_downsampled.png --scale_factor 0.35 # model 4
python train.py --input_image balloons_downsampled.png --alpha 1 # model 5
python train.py --input_image balloons_downsampled.png --number_of_noises 3 # model 6



# Models trained on pyramids
python train.py --input_image pyramids_downsampled.jpg # model 1
python train_patches.py --input_image pyramids_downsampled.jpg # model 2

python train.py --input_image pyramids_downsampled.jpg --number_of_noises 12 # model 3
python train.py --input_image pyramids_downsampled.jpg --scale_factor 0.35 # model 4
python train.py --input_image pyramids_downsampled.jpg --alpha 1 # model 5