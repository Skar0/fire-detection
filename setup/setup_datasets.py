import os

"""
This module contains functions which allow to download and setup the datasets used for this project.
"""


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


size_rep_dict = {'small': 0, 'medium': 3, 'big': 4}


def necessary_setup_fire_detection(size='small'):
    datasets_path = 'datasets' + '/' + size
    fire = not (os.path.exists(datasets_path + '/fire') and
                len(os.listdir(datasets_path + '/fire')) != 0)
    no_fire = not (os.path.exists(datasets_path + '/no_fire') and
                   len(os.listdir(datasets_path + '/no_fire')) != 0)
    start_fire = not (os.path.exists(datasets_path + '/start_fire') and
                      len(os.listdir(datasets_path + '/start_fire')) != 0)

    return fire and no_fire and start_fire


def download_and_setup_dataset_fire_detection(size='small', verbose=0):  # size in {'small', 'medium', 'big'}


    def verprint(s):
        if verbose:
            print(s, flush=True)

    necessary = necessary_setup_fire_detection(size)

    if necessary:

        if size != 'all':
            # get number of repetitions
            rep = size_rep_dict[size]

            prefix = 'wget https://github.com/belarbi2733/keras_yolov3/releases/download/1/'
            inter_command = 'defi1certif-datasets-fire_' + size + '.tar'

            # get dataset
            if rep:
                for i in range(1, rep + 1):
                    suffix = '.00' + str(i)

                    command = prefix + inter_command + suffix
                    verprint('+ executing: ' + command)
                    os.system(command)

                # recombine when needed
                recombine_command = 'cat'
                for i in range(1, rep + 1):
                    suffix = '.00' + str(i)

                    recombine_command += ' ' + inter_command + suffix

                recombine_command += ' > ' + inter_command

                verprint('+ executing: ' + recombine_command)
                os.system(recombine_command)

                # clean behind you
                clean_command = 'rm'
                for i in range(1, rep + 1):
                    suffix = '.00' + str(i)
                    clean_command += ' ' + inter_command + suffix
                verprint('+ executing: ' + clean_command)
                os.system(clean_command)

            else:
                verprint('+ executing: ' + prefix + inter_command)
                os.system(prefix + inter_command)

            # created dirs
            datasets_path = "datasets"
            verprint("- attempting to create 'datasets' directory")
            if os.path.exists(datasets_path) == False:
                verprint("- creating 'datasets' directory")
                os.makedirs(datasets_path)
            else:
                verprint("- 'datasets' directory already exists")

            # put the each dataset in its asociate folder
            prefix = 'tar xf '
            suffix = " -C 'datasets' --one-top-level && mv datasets/defi1certif-datasets-fire_" + size + " datasets/" + size

            # execute
            command = prefix + inter_command + suffix
            verprint('+ executing: ' + command)
            os.system(command)

            # clean behind you
            if os.path.exists("datasets/defi1certif-datasets-fire_" + size):
                verprint('+ executing: ' + 'rm -r datasets/defi1certif-datasets-fire_' + size)
                os.system('rm -r datasets/defi1certif-datasets-fire_' + size)

            verprint('- ' + size + ' dataset successfully setup')

        else:
            for key in size_rep_dict:
                download_and_setup_dataset_fire_detection(size=key)


def setup_full_dataset_fire_detection(verbose=0):
    """
    Combines all datasets in a single folder.
    :return:
    """

    def verprint(s):
        if verbose:
            print(s, flush=True)

    verprint('- fusioning all datasets', flush=True)

    verprint('- creating directories', flush=True)
    # creating the folder to merge datasets
    if not os.path.exists("datasets/all"):
        os.makedirs("datasets/all")
    if not os.path.exists("datasets/all/fire"):
        os.makedirs("datasets/all/fire")
    if not os.path.exists("datasets/all/no_fire"):
        os.makedirs("datasets/all/no_fire")
    if not os.path.exists("datasets/all/start_fire"):
        os.makedirs("datasets/all/start_fire")

    verprint('- moving files', flush=True)
    # moving images from the small dataset to the full dataset
    verprint('+ executing: ' + "find datasets/small/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/",
             flush=True)
    os.system("find datasets/small/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/")

    verprint('+ executing: ' + "find datasets/small/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/",
             flush=True)
    os.system("find datasets/small/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/")

    verprint(
        '+ executing: ' + "find datasets/small/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/",
        flush=True)
    os.system("find datasets/small/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/")

    # moving images from the medium dataset to the full dataset
    verprint('+ executing: ' + "find datasets/medium/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/",
             flush=True)
    os.system("find datasets/medium/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/")

    verprint('+ executing: ' + "find datasets/medium/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/",
             flush=True)
    os.system("find datasets/medium/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/")

    verprint(
        '+ executing: ' + "find datasets/medium/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/",
        flush=True)
    os.system("find datasets/medium/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/")

    # moving images from the large dataset to the full dataset
    verprint('+ executing: ' + "find datasets/big/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/", flush=True)
    os.system("find datasets/big/fire -type f -print0 | xargs -0 mv -t datasets/all/fire/")

    verprint('+ executing: ' + "find datasets/big/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/",
             flush=True)
    os.system("find datasets/big/no_fire -type f -print0 | xargs -0 mv -t datasets/all/no_fire/")

    verprint('+ executing: ' + "find datasets/big/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/",
             flush=True)
    os.system("find datasets/big/start_fire -type f -print0 | xargs -0 mv -t datasets/all/start_fire/")

    verprint("- files moved to:'datasets/all/", flush=True)

    space = len("- files moved to:'datasets/all/")
    verprint(' ' * space + "fire'", flush=True)
    verprint(' ' * space + "no_fire'", flush=True)
    verprint(' ' * space + "start_fire'", flush=True)

    verprint('- done', flush=True)


def download_and_setup_full_dataset_fire_detection():
    download_and_setup_dataset_fire_detection('full')
    setup_full_dataset_fire_detection()
