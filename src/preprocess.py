from keras.utils import image_dataset_from_directory
from config import train_directory, test_directory, image_size, batch_size, validation_split
from config import transfer_train_directory, transfer_test_directory, transfer_batch_size

def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset

def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    return train_dataset, validation_dataset, test_dataset

def get_transfer_datasets():
    # Uses transfer_batch_size instead of the shared batch_size, since the
    # transfer dataset is much smaller and needs smaller batches for
    # enough gradient updates per epoch
    train_dataset, validation_dataset, test_dataset = \
        _split_data(transfer_train_directory, transfer_test_directory, transfer_batch_size, validation_split)
    return train_dataset, validation_dataset, test_dataset
