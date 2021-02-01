import config
from preprocess_data import load_train_csv_bb, load_train_dataset, read_image_as_array

filenames = []
bounding_box = []
images = []

def load_img():
    img = []

    filenames_dataset = load_train_dataset(config.train_dir)
    for file in filenames_dataset:
        image = read_image_as_array(config.train_dir + file)
        img.append(image)

    return img

if __name__ == "__main__":

    #load filenames and bounding boxes
    filenames, bounding_box = load_train_csv_bb(config.train_csv)

    #load images
    images = load_img()


