import glob
import numpy as np
import random
from PIL import Image
# from keras.preprocessing.image import ImageDataGenerator

ALL = 11200
TEST = 2240

arr_maze = []
arr_pig = []
maze_files = glob.glob("data/augmentation/output/maze/*.png")
pig_files = glob.glob("data/augmentation/output/pig/*.png")
for i in range(ALL / 2):
    img = Image.open(maze_files[i])
    arr_maze.append((np.array(img), 0))
    print("Reading %s images" % str(i + 1))
    img.close()

for i in range(ALL / 2):
    img = Image.open(pig_files[i])
    arr_pig.append((np.array(img), 1))
    print("Reading %s images" % str(i + ALL / 2 + 1))
    img.close()

training_set = []
test_set = []
training_label = []
test_label = []
# random.Random().shuffle(arr)
m = 0
n = 0
for i in range(ALL):
    print("Save %s images" % str(i + 1))
    rand = random.randint(1, 10)
    if rand == 4 or rand == 8:
        if len(test_set) < TEST:
            if random.randint(0, 1) == 0 and m < ALL / 2:
                test_set.append(arr_maze[m][0])
                test_label.append(arr_maze[m][1])
                m += 1
            elif n < ALL / 2:
                test_set.append(arr_pig[n][0])
                test_label.append(arr_pig[n][1])
                n += 1
        else:
            if random.randint(0, 1) == 0 and m < ALL / 2:
                training_set.append(arr_maze[m][0])
                training_label.append(arr_maze[m][1])
                m += 1
            elif n < ALL / 2:
                training_set.append(arr_pig[n][0])
                training_label.append(arr_pig[n][1])
                n += 1
    else:
        if len(training_set) < ALL - TEST:
            if random.randint(0, 1) == 0 and m < ALL / 2:
                training_set.append(arr_maze[m][0])
                training_label.append(arr_maze[m][1])
                m += 1
            elif n < ALL / 2:
                training_set.append(arr_pig[n][0])
                training_label.append(arr_pig[n][1])
                n += 1
        else:
            if random.randint(0, 1) == 0 and m < ALL / 2:
                test_set.append(arr_maze[m][0])
                test_label.append(arr_maze[m][1])
                m += 1
            elif n < ALL / 2:
                test_set.append(arr_pig[n][0])
                test_label.append(arr_pig[n][1])
                n += 1

for i in range(ALL - TEST):
    img = Image.fromarray(np.array(training_set[i]))
    img.save("data/augmentation/split/training/training_" + str(i) + ".png")

for i in range(TEST):
    img = Image.fromarray(np.array(test_set[i]))
    img.save("data/augmentation/split/test/test_" + str(i) + ".png")

# datagen = ImageDataGenerator(featurewise_center=True,
#                              featurewise_std_normalization=True)

# datagen.fit(training_set)

# training = datagen.flow(np.array(training_set), None,
#                         batch_size=(ALL - TEST), shuffle=False,
#                         save_to_dir="record/images/preprocessed/training/",
#                         save_prefix="training")

# test = datagen.flow(np.array(test_set), None,
#                     batch_size=TEST, shuffle=False,
#                     save_to_dir="record/images/preprocessed/test/",
#                     save_prefix="test")

# training.next()

# test.next()

file_training = "data/augmentation/split/training_label.txt"
file_test = "data/augmentation/split/test_label.txt"
with open(file_training, "w") as ftr:
    for i in range(ALL - TEST):
        ftr.write(str(training_label[i]) + "\n")
ftr.close()
with open(file_test, "w") as ft:
    for i in range(TEST):
        ft.write(str(test_label[i]) + "\n")
ft.close()
