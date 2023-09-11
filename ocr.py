from PIL import Image
import pytesseract
import os, time
from multiprocessing import Pool

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

    with open(path_to_save + f"_{start_idx}.txt", "w") as f:
        f.write(ocr_text)

    return ocr_text

    


def worker(num):
    """Simple function to do some work"""
    print(f'Worker {num} is working')

if __name__ == '__main__':
    # Create a pool of processes
    start_time = time.time()

    number_of_images = get_number_of_images("dragon")
    batch_start_indices = get_indices_of_start_batch(number_of_images)

    num_processes = 4
    pool = Pool(processes=num_processes)

    for i in range(len(batch_start_indices) - 1):
        print(f"Started executing batch {i+1}")
        start_idx = batch_start_indices[i]
        end_idx = batch_start_indices[i + 1]
        pool.apply_async(save_ocr_text, args=("dragon", start_idx, end_idx))

    pool.close()
    pool.join() 

    print(f"Time taken: {time.time() - start_time} seconds")
