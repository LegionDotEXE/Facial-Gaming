categories = ['neutral', 'happy', 'surprise']

train_directory = 'train'
test_directory = 'test'

train_size = 5000
original_image_size = (48, 48)
image_size = (150, 150)
batch_size = 128
validation_split = 0.2

BOARD_SIZE = 3

# Section 8: Transfer Learning
# way as train/test above (one subfolder per class, e.g. transfer_train/dog, transfer_train/cat)
transfer_categories = ['dogs', 'cats']
transfer_train_directory = 'transfer_train_final'
transfer_test_directory = 'transfer_test_final'
transfer_batch_size = 32
