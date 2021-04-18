import os
import config
from tensorflow.keras.models import load_model
from preprocess_data_bb import data_generator


def evaluate(model_evaluate):
    test_data_gen = data_generator(config.test_dir, config.test_csv, config.batch_size)

    print("Batch size =", config.batch_size)

    files_test = [f for f in os.listdir(config.test_dir) if os.path.isfile(os.path.join(config.test_dir, f))]
    test_steps = (len(files_test) // config.batch_size) + 1
    print("Test steps =", test_steps)

    results = model_evaluate.evaluate(x=test_data_gen, verbose=1, return_dict=True, steps=test_steps)
    loss = results['loss']
    print('\nMODEL: {}\nLOSS: {:.4f}'.format(config.model_name, loss))


if __name__ == "__main__":
    if os.path.isfile(config.model_detector):
        print("Loading object detector...")
        model = load_model(config.model_detector)

        evaluate(model)
    else:
        print("Model not found in {}".format(config.model_detector))
