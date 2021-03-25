working_dir = "./"

# file
file_dir = working_dir + 'file/'
#labels_file = file_dir + 'text_mined_labels_171_and_split.json'
#labels_train = file_dir + 'finalLabelsTraining.txt'
#labels_val = file_dir + 'finalLabelsValidation.txt'
train_csv = file_dir + 'Train_info.csv'
test_csv = file_dir + 'Test_info.csv'
val_csv = file_dir + 'Val_info.csv'

# image
dataset_dir = working_dir + 'dataset/'
train_dir = dataset_dir + 'train/'
test_dir = dataset_dir + 'test/'
val_dir = dataset_dir + 'val/'

# image BB
dataset_dir = working_dir + 'dataset/prova/'
train_dir_bb = dataset_dir + 'train'
test_dir_bb = dataset_dir + 'test'
val_dir_bb = dataset_dir + 'validation'

# model
models_dir = working_dir + 'models'
model_name = "VGG16"
model_detector = models_dir + "/VGG16_detector.h5"
train_log_dir = working_dir + 'training_logs/'
train_log_file = train_log_dir + model_name + "_log.csv"
plot_path_loss = train_log_dir + "VGG16_plot_loss.png"
plot_path_acc = train_log_dir + "VGG16_plot_accuracy.png"

batch_size = 32
total_epochs = 25
