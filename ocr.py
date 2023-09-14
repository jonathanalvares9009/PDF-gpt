from PIL import Image
import pytesseract
import os
from multiprocessing import Pool
from transformers import pipeline

def get_number_of_images(folder_name):
    """Returns the number of images in a directory"""
    path = f"output/images/{folder_name}"
    return len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])

def get_indices_of_start_batch(num_images):
    """Returns a list of indices to start the batch. Each batch should have 10 or less elements"""
    indices = []
    for i in range(0, num_images, 10):
        indices.append(i)
    indices.append(num_images)
    return indices

def save_ocr_text(folder_name, start_idx, end_idx):
    """It extracts the text from the images in the path and saves it to a file"""
    path = f"output/images/{folder_name}/"

    ocr_text = ""
    for i in range(start_idx, end_idx):
        img = Image.open(path + f"_{i}.jpg")
        ocr_text += pytesseract.image_to_string(img)

    path_to_save = f"output/text/{folder_name}/"

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    with open(path_to_save + f"_{int(start_idx / 10) + 1}.txt", "w") as f:
        f.write(ocr_text)

    return ocr_text

def process_images(folder_name):
    """It processes the images in the folder and saves the text to a file"""
    if not os.path.exists(f"output/images/{folder_name}"):
        return
    
    if os.path.exists(f"output/text/{folder_name}"):
        return

    number_of_images = get_number_of_images(folder_name)
    batch_start_indices = get_indices_of_start_batch(number_of_images)

    pool = Pool()

    for i in range(len(batch_start_indices) - 1):
        start_idx = batch_start_indices[i]
        end_idx = batch_start_indices[i + 1]
        pool.apply_async(save_ocr_text, args=(folder_name, start_idx, end_idx))

    pool.close()
    pool.join() 

if __name__ == "__main__":
    image = Image.open("/Users/jonathanalvares/Downloads/IMG_1422.jpg")
    pipe = pipeline("image-to-text", model="DunnBC22/trocr-base-handwritten-OCR-handwriting_recognition_v2")

    
    # print(pipe(image))
