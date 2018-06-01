import Augmentor

p = Augmentor.Pipeline("data/augmentation/")

# p.skew_tilt(probability=1)

# p.skew_left_right(probability=1)

# p.skew_top_bottom(probability=1)

# p.skew_corner(probability=1)

# p.random_distortion(probability=1, grid_width=10, grid_height=10,
#                      magnitude=8)

# p.rotate90(probability=1)

# p.rotate180(probability=1)

# p.rotate270(probability=1)

# p.shear(probability=1, max_shear_left=25, max_shear_right=25)

# p.flip_left_right(probability=1)

# p.flip_top_bottom(probability=1)

p.crop_random(probability=1, percentage_area=0.8)
p.resize(probability=1, width=128, height=128)

p.process()
