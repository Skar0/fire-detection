import argparse

from custom_model.cladoh import Cladoh, preprocess_input_custom
from setup.setup_datasets import *
from transfer_learning import *

whole_printer = 0

def train_and_save_model(dataset_path, percentage=0.8, nbr_epochs=10, batch_size=32):
    """
    :param percentage: percentage of samples to be used for training. Must be in [0,1].
    :param nbr_epochs:
    :param batch_size:
    """

    custom_based_model_save_folder = "model-saves/custom_based/"

    # create save path
    if not os.path.exists(custom_based_model_save_folder):
        os.makedirs(custom_based_model_save_folder)

    custom_based_model_save_path = custom_based_model_save_folder + "custom_trained_save.h5"

    # if a model already exists we load it, weights should be better
    # use (load|save)_weights function always.

    # old_model:
    if os.path.exists(custom_based_model_save_path):
        model = load_model(custom_based_model_save_path)
    else:
        model = Cladoh(include_top=True, pooling='max', input_shape=(224, 224, 3))

    # checkpoints

    # We can do learning rate adaptation later as part of fine tuning or use adaptive optimizer (rmsprop, adam)
    # keras.callbacks.callbacks.LearningRateScheduler(schedule, verbose=0)
    # keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
    # min_delta=0.0001, cooldown=0, min_lr=0)

    # saves the model when validation accuracy improves
    save_on_improve = ModelCheckpoint(custom_based_model_save_path, monitor='val_accuracy', verbose=1,
                                      save_best_only=True, save_weights_only=False, mode='max')

    # EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',baseline=None, res
    # tore_best_weights=False)

    """
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                              write_grads=False, write_images=False, embeddings_freq=0,
                              embeddings_layer_names=None, embeddings_metadata=None,
                              embeddings_data=None, update_freq='epoch')
    callbacks = [save_on_improve, tensorboard]
    """

    # loss is categorical since we are classifying
    model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'], )
    # callbacks=callbacks)

    (train_samples, train_labels), (val_samples, val_labels) = extract_dataset(dataset_path, classes, percentage)

    if whole_printer:
        print(train_samples.shape)
        print(train_labels.shape)
        print(val_samples.shape)
        print(val_labels.shape)

    training_sample_generator = augmented_batch_generator(train_samples,
                                                          train_labels,
                                                          batch_size,
                                                          preprocess_input_custom,
                                                          True,
                                                          image_size=(224, 224, 3))

    validation_sample_generator = augmented_batch_generator(val_samples,
                                                            val_labels,
                                                            batch_size,
                                                            preprocess_input_custom,
                                                            True,
                                                            image_size=(224, 224, 3))

    nbr_train_samples = len(train_samples)
    nbr_val_samples = len(val_samples)

    # call to fit using a generator
    history = model.fit_generator(
        generator=training_sample_generator,
        steps_per_epoch=math.ceil(nbr_train_samples / batch_size),
        epochs=nbr_epochs,
        validation_data=validation_sample_generator,
        validation_steps=math.ceil(nbr_val_samples / batch_size),
        verbose=1)

    model.save(custom_based_model_save_path)

    print('model saved to: ', custom_based_model_save_path)
    return custom_based_model_save_path, history


if __name__ == '__main__':
    # some initial values

    classes = ['fire', 'no_fire', 'start_fire']
    nbr_classes = 3
    classes_value = classes

    """
    classes_value = classes
    split_percentage = 0.8  # @param {type:"slider", min:0.3, max:0.9, step:0.1}

    nbr_batch_size = 64  # @param [1,2,4,8,16,32,64,128, 256] {type:"raw"}
    dataset_name = 'big'  # @param ["small","medium","big","all"]

    dataset_path = os.path.join('datasets/', dataset_name)
    epochs = 30  # @param {type:"slider", min:5, max:100, step:5}
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--split_percentage', default=0.8, type=float)
    parser.add_argument('--dataset', default='small', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=10, type=int)

    args = parser.parse_args()

    split_percentage = args.split_percentage
    nbr_batch_size = args.batch_size
    dataset_name = args.dataset
    epochs = args.epochs

    dataset_path = os.path.join('datasets/', dataset_name)

    print('nbr_classes: ', nbr_classes)
    print('classes_value: ', classes_value)
    print('split_percentage: ', split_percentage)
    print('nbr_batch_size: ', nbr_batch_size)
    print('dataset_name: ', dataset_name)
    print('dataset_path: ', dataset_path)
    print('epochs: ', epochs)

    # download and setup dataset
    if not os.path.exists(dataset_path):
        download_and_setup_dataset_fire_detection(dataset_name)

    # remove anything problematic
    os.system('rm -r ' + dataset_path + '/de*')

    # define, train and save custom model
    # this function returns the path where the model has been saved to

    model_path, history = train_and_save_model(dataset_path, percentage=split_percentage, nbr_epochs=epochs,
                                                      batch_size=nbr_batch_size)

