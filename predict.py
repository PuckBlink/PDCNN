
from PIL import Image

from drcnn import DRCNN
import os


if __name__ == "__main__":
    drcnn = DRCNN()

    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    img_name = '1.jpg'

    image_path = os.path.join(dir_origin_path, img_name)
    image = Image.open(image_path)
    r_image = drcnn.detect_image(image)
    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)
    r_image.save(os.path.join(dir_save_path, img_name))
