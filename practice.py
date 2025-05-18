import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np
    import os
    import random
    from PIL import Image
    from glob import glob
    import cv2
    import tensorflow as tf
    from tqdm import tqdm
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.optimizers import Adam
    return (
        Adam,
        Conv2D,
        Dense,
        Flatten,
        MaxPooling2D,
        Sequential,
        cv2,
        glob,
        mo,
        np,
        os,
        pl,
        plt,
        shuffle,
        tqdm,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(
        f'''
        # Running CNN on Musical Instrument Dataset
        '''
    )
    return


@app.cell
def _(mo):
    mo.md(f"## Getting data")
    return


@app.cell
def _():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("nikolasgegenava/music-instruments")

    print("Path to dataset files:", path)
    return (path,)


@app.cell
def _(mo):
    mo.md(r"""## List all the files""")
    return


@app.cell
def _(os, path, pl):
    def list_files(startpath):
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))

    stat_df = pl.read_csv(os.path.join(path, "music_instruments", "dataset_stats.csv"))
    stat_df.head()
    return


@app.cell
def _(mo):
    mo.md(r"""## Plot Images""")
    return


@app.cell
def _(cv2, glob, os, path, plt):
    def plotImages(tools, directory):
        print(tools)
        multipleImages = glob(directory)
        plt.rcParams['figure.figsize'] = (15, 15)
        plt.subplots_adjust(wspace=0, hspace=0)
        i_ = 0
        for l in multipleImages[:25]:
            im = cv2.resize(cv2.imread(l), (128, 128))
            plt.subplot(5,5,i_+1)
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            i_+=1
        plt.show()

    plotImages("Flute images", os.path.join(path, "music_instruments", "flute/**"))
    return


@app.cell
def _(os, path, pl):
    df = pl.read_csv(os.path.join(path,'music_instruments/dataset_stats.csv'), encoding='utf-8')
    df.tail()
    return


@app.cell
def _(os, path):
    # Directories for each musical instrument category
    data_dirs = {    
        "accordion": os.path.join(path, "music_instruments/accordion"),
        "banjo": os.path.join(path, "music_instruments/banjo"),
        "drum": os.path.join(path, "music_instruments/drum"),
        "flute": os.path.join(path, "music_instruments/flute"),
        "guitar": os.path.join(path, "music_instruments/guitar"),
        "harmonica": os.path.join(path, "music_instruments/harmonica"),
        "saxophone": os.path.join(path, "music_instruments/saxophone"),
        "sitar": os.path.join(path, "music_instruments/sitar"),
        "tabla": os.path.join(path, "music_instruments/tabla"),
        "violin": os.path.join(path, "music_instruments/violin"),
    }
    return (data_dirs,)


@app.cell
def _(mo):
    mo.md(f"## Show images from each directory")
    return


@app.cell
def _(cv2, data_dirs, os, plt):
    # Function to load and display images from each directory
    def show_sample_images():
        for label, dir_path in data_dirs.items():
            if not os.path.exists(dir_path):
                print(f"Warning: Directory {dir_path} does not exist.")
                continue

            # Load the first few images from the directory
            sample_images = []
            for root, _, files in os.walk(dir_path):
                for img_name in files[:5]:  # Load 5 images for display
                    img_path = os.path.join(root, img_name)
                    if img_path.lower().endswith(('.JPG', '.jpg', '.jpeg', 'png')):
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.resize(img, (150, 150))
                            sample_images.append(img)
                        if len(sample_images) == 5:
                            break
                if len(sample_images) == 5:
                    break
                
            # Plot the images
            plt.figure(figsize=(10, 10))
            for i, img in enumerate(sample_images):
                plt.subplot(1, 5, i + 1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(f"{label}")
            plt.show()

    # Call the function to display images
    show_sample_images()
    return


@app.cell
def _(mo):
    mo.md(f"## Loading images and labels to train the Conv2D Network")
    return


@app.cell
def _(cv2, data_dirs, np, os, shuffle, tqdm, train_test_split):
    IMG_SIZE = 150
    X, Y = [], []

    # function to load images and labels
    def load_images():
        for label, dir_path in data_dirs.items():
            if not os.path.exists(dir_path):
                print(f"Warning: Directory {dir_path} does not exist.")
                continue

            for root, _, files in os.walk(dir_path):  # Walk through all subdirectories
                for img_name in tqdm(files):
                    img_path = os.path.join(root, img_name)

                    # Ensure it's a valid image file
                    if not img_path.lower().endswith(('.JPG', '.jpg', '.jpeg', 'png')):
                        print(f"Skipping {img_path}, not a valid image file.")
                        continue

                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Could not read {img_path}. Skipping.")
                        continue  # Skip unreadable images

                    try:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        X.append(img)
                        Y.append(label)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")

    # Load the dataset
    load_images()

    # Convert to numpy arrays
    if len(X) == 0 or len(Y) == 0 or len(X) != len(Y):
        raise ValueError(f"Mismatch in dataset sizes: X={len(X)}, Y={len(Y)}")

    X = np.array(X, dtype='float32') / 255.0  # Normalize images
    Y = np.array([list(data_dirs.keys()).index(y) for y in Y], dtype='int32')

    # Shuffle and split dataset
    X, Y = shuffle(X, Y, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print(f"Dataset successfully loaded: Train={len(X_train)}, Test={len(X_test)}")
    return IMG_SIZE, X_test, X_train, Y_test, Y_train


@app.cell
def _(mo):
    mo.md(f"## Build and compile the model")
    return


@app.cell
def _(
    Adam,
    Conv2D,
    Dense,
    Flatten,
    IMG_SIZE,
    MaxPooling2D,
    Sequential,
    data_dirs,
):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(data_dirs), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return (model,)


@app.cell
def _(mo):
    mo.md(f"## Train and evaluate")
    return


@app.cell
def _(X_train, Y_train, model):
    # Train the model
    history = model.fit(X_train, Y_train, batch_size=32, epochs=2, validation_split=0.2) #Epochs=10
    return (history,)


@app.cell
def _(X_test, Y_test, history, model, plt):
    # Plot training history
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.legend()
    plt.title("Model Accuracy")
    plt.show()

    # Evaluate on test data
    loss, acc = model.evaluate(X_test, Y_test)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    return


if __name__ == "__main__":
    app.run()
