import os


def download_and_setup_test_dataset():
    """
    Downloading the test dataset.
    """
    os.system(
        'wget https://github.com/belarbi2733/keras_yolov3/releases/download/1/test_database.tar')

    datasets_path = "datasets"
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)

    # put the test dataset in datasets/test
    os.system("tar xf test_database.tar -C 'datasets' --one-top-level && mv test_database.tar datasets/test")


def download_and_setup_small_dataset():
    """
    Downloading the small dataset.

    """
    os.system(
        'wget https://github.com/belarbi2733/keras_yolov3/releases/download/1/defi1certif-datasets-fire_small.tar')

    datasets_path = "datasets"
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)

    # put the small dataset in datasets/small
    os.system(
        "tar xf defi1certif-datasets-fire_small.tar -C 'datasets' --one-top-level && mv "
        "datasets/defi1certif-datasets-fire_small datasets/small")


def download_and_setup_medium_dataset():
    """
    Downloading the medium dataset.
    """
    os.system(
        'wget https://github.com/belarbi2733/keras_yolov3/releases/download/1/defi1certif-datasets-fire_medium.tar.001')
    os.system(
        'wget https://github.com/belarbi2733/keras_yolov3/releases/download/1/defi1certif-datasets-fire_medium.tar.002')
    os.system(
        'wget https://github.com/belarbi2733/keras_yolov3/releases/download/1/defi1certif-datasets-fire_medium.tar.003')

    datasets_path = "datasets"
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)

    # recombine the tar files
    os.system("cat  defi1certif-datasets-fire_medium.tar.001 defi1certif-datasets-fire_medium.tar.002 "
              "defi1certif-datasets-fire_medium.tar.003 >> defi1certif-datasets-fire_medium.tar")

    # put the medium dataset in datasets/medium
    os.system("tar xf defi1certif-datasets-fire_medium.tar -C 'datasets' --one-top-level && mv "
              "datasets/defi1certif-datasets-fire_medium datasets/medium")


def download_and_setup_large_dataset():
    """
    Downloading the large dataset.
    """
    os.system(
        'wget https://github.com/belarbi2733/keras_yolov3/releases/download/1/defi1certif-datasets-fire_big.tar.001')
    os.system(
        'wget https://github.com/belarbi2733/keras_yolov3/releases/download/1/defi1certif-datasets-fire_big.tar.002')
    os.system(
        'wget https://github.com/belarbi2733/keras_yolov3/releases/download/1/defi1certif-datasets-fire_big.tar.003')
    os.system(
        'wget https://github.com/belarbi2733/keras_yolov3/releases/download/1/defi1certif-datasets-fire_big.tar.004')

    datasets_path = "datasets"
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)

    # recombine the tar files
    os.system("cat  defi1certif-datasets-fire_big.tar.001 defi1certif-datasets-fire_big.tar.002 "
              "defi1certif-datasets-fire_big.tar.003 defi1certif-datasets-fire_big.tar.004 >> "
              "defi1certif-datasets-fire_big.tar")

    # put the large dataset in datasets/large
    os.system("tar xf defi1certif-datasets-fire_big.tar -C 'datasets' --one-top-level && mv "
              "datasets/defi1certif-datasets-fire_big datasets/large")


def setup_full_dataset():
    """
    Downloads and sets up all datasets in a single folder named all.
    A folder per class is created.
    """
    download_and_setup_small_dataset()
    download_and_setup_medium_dataset()
    download_and_setup_large_dataset()

    # creating the folder to merge datasets
    if not os.path.exists("datasets/all"):
        os.makedirs("datasets/all")
    if not os.path.exists("datasets/all/fire"):
        os.makedirs("datasets/all/fire")
    if not os.path.exists("datasets/all/no_fire"):
        os.makedirs("datasets/all/no_fire")
    if not os.path.exists("datasets/all/start_fire"):
        os.makedirs("datasets/all/start_fire")

    # moving images from the small dataset to the full dataset
    os.system("find datasets/small/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/")
    os.system("find datasets/small/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/")
    os.system("find datasets/small/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/")

    # moving images from the medium dataset to the full dataset
    os.system("find datasets/medium/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/")
    os.system("find datasets/medium/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/")
    os.system("find datasets/medium/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/")

    # moving images from the large dataset to the full dataset
    os.system("find datasets/large/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/")
    os.system("find datasets/large/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/")
    os.system("find datasets/large/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/")
