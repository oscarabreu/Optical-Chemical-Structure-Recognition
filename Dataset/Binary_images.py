import os
from PIL import Image
from tqdm import tqdm

# Define source and output directories
src = './images'
out = './binary_images'

# Ensure output directory exists
if not os.path.exists(out):
    os.makedirs(out)

# Iterate over each image in the source directory
for image in tqdm(os.listdir(src)):
    image_path = os.path.join(src, image)
    image_out_path = os.path.join(out, image)

    # Open the image
    img = Image.open(image_path)

    # Convert the image to grayscale ('L' mode stands for luminance)
    gray = img.convert('L')

    # Convert grayscale image to binary (black and white) format
    # Pixels with a value below 200 are turned black, others are white
    binary_image = gray.point(lambda x: 0 if x < 200 else 255, '1')

    # Save the binary image
    binary_image.save(image_out_path)

    # Reopen the binary image, convert it to grayscale and save it
    img_255 = Image.open(image_out_path)
    img_255 = img_255.convert('L')
    img_255.save(image_out_path)