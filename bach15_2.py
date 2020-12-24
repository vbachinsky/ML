import keras.models as keras_models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras.preprocessing as keras_preprocessing


np.set_printoptions(linewidth=200)

image_file0 = "data/0.bmp"
image_file1 = "data/1.bmp"
image_file2 = "data/2.bmp"
image_file3 = "data/3.bmp"
image_file4 = "data/4.bmp"
image_file5 = "data/5.bmp"
image_file7 = "data/7.bmp"
image_file8 = "data/8.bmp"

model = keras_models.load_model("models/MNIST_CNN.h5")
image_files = (image_file0, image_file1, image_file2, image_file3, image_file4, image_file5, image_file7, image_file8)
fig, axs = plt.subplots(len(image_files), 1, figsize=(10, 10))

for image_file in range(len(image_files)):
    # Plot the numbers
    img = cv2.imread(image_files[image_file])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    axs[image_file].imshow(img, cmap="Greys")

    # Prediction
    img = keras_preprocessing.image.load_img(image_files[image_file], color_mode="grayscale",
                                             target_size=[28, 28], interpolation="nearest")

    input_arr = keras_preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])
    input_arr = 1 - input_arr/255.0

    predictions_single = model.predict(input_arr)
    prediction_number = np.argmax(predictions_single[0])

    axs[image_file].set_xlabel(f"Predicted number: {prediction_number}")
    print(predictions_single)
    print("Predicted number: {}".format(prediction_number))

fig.tight_layout()
plt.show()
