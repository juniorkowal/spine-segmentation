import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torchio as tio
from data_transforms.classes import *
from dataloader import VerSe
from KDEpy import FFTKDE
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm
from utils.constants import remapping_cervical_thoraic_lumbar


def get_histogram(tensor, num_positions=100):
    values = tensor.numpy().ravel()

    bandwidth = 'ISJ'
    kde = FFTKDE(kernel='gaussian', bw=bandwidth)

    # kernel = stats.gaussian_kde(values)
    # positions = np.linspace(values.min(), values.max(), num=num_positions)
    # histogram = kernel(positions)
    positions, histogram = kde.fit(values).evaluate(num_positions)
    return positions, histogram


def plot_spacing_histogram(spacings, save=False, show_plot = False):
    dataset_name = os.path.basename(save)
    if 'standardized' in dataset_name:
        save = save.split('_standardized')[0]
    spacing_counts = Counter(map(tuple, spacings))
    sorted_spacings = sorted(spacing_counts.items())
    tuples, counts = zip(*sorted_spacings)

    tuple_labels = [f"{t}" for t in tuples]

    fig, ax = plt.subplots()
    bars = ax.bar(tuple_labels, counts, alpha=0.7)
    ax.set_xlabel('Spacing Tuples (rounded)')
    ax.set_ylabel('Count')
    ax.set_title(f'Spacing Histogram for {dataset_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    if save:
        filename = f"{save}/spacing_histogram_{dataset_name}.png"
        plt.savefig(filename)
        print(f"Saved spacing histogram as {filename}")

    if show_plot:
        plt.show()
    plt.close()


def plot_label_counts(labels_list, remapping=None, save=False, show_plot = False):
    dataset_name = os.path.basename(save)
    if 'standardized' in dataset_name:
        save = save.split('_standardized')[0]
    label_counts = Counter(np.concatenate(labels_list))
    labels, counts = zip(*label_counts.items())

    color_map = {1: 'red', 2: 'blue', 3: 'green'}
    colors = [color_map.get(remapping[label], 'blue') if remapping else 'blue' for label in labels]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, counts, alpha=0.7, color = colors)
    ax.set_xlabel('Klasa')
    ax.set_ylabel('Ilość')
    # ax.set_title(f'Label Count Histogram for {dataset_name}')
    ax.set_title('Rozkład klas w zbiorze danych')
    # Determine the range of class labels, excluding 0
    min_label = int(min(labels))
    max_label = int(max(labels))
    xticks = list(range(min_label, max_label + 1))

    # Remove 0 from the ticks if it exists
    if 0 in xticks:
        xticks.remove(0)

    # Set the x-axis ticks
    ax.set_xticks(xticks)

    # Optionally rotate the x-tick labels for better readability
    # plt.xticks(rotation=90)

    # plt.xticks(rotation=90)
    plt.tight_layout()

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    if save:
        filename = f"{save}/label_count_{dataset_name}.png"
        plt.savefig(filename)
        print(f"Saved label counts histogram as {filename}")

    if show_plot:
        plt.show()
    plt.close()


def plot_histograms(histogram_list, save=False, show_plot=False):
    # Extract dataset name from the file path
    dataset_name = os.path.basename(save)
    if 'standardized' in dataset_name:
        save = save.split('_standardized')[0]

    # Plot each histogram in the list
    for hist in histogram_list:
        plt.plot(hist[0], hist[1], color='black', alpha=0.05)

    # Set the title and axis labels
    # plt.title(f'Image histograms for {dataset_name}')
    plt.title(f'Histogramy obrazów TK zbioru danych')

    plt.xlabel('Wartości wokseli w HU')
    plt.ylabel('Ilość')

    # Format the y-axis to use scientific notation
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Adjust layout for better fit
    plt.tight_layout()

    # Save the plot if required
    if save:
        filename = f"{save}/histograms_{dataset_name}.png"
        plt.savefig(filename)
        print(f"Saved histograms as {filename}")

    # Show the plot if required
    if show_plot:
        plt.show()

    # Close the plot to free up memory
    plt.close()


def get_verse_dataset_statistics(root, edition, download, histogram_standardization, show = False):
    transform = tio.Clamp(out_min=-1024)
    transform = None
    train_set = VerSe(root=root, split='training', edition=edition, download=download, transform=transform)
    val_set = VerSe(root=root, split='validation', edition=edition, download=download, transform=transform)
    test_set = VerSe(root=root, split='test', edition=edition, download=download, transform=transform)

    histogram_transform = None

    if histogram_standardization:
        output_path = f'{train_set.root}/histogram_standardization.npy'

        if os.path.exists(output_path):
            print(f"\nHistogram standardization file already exists at {output_path}.\nSkipping training histogram standardization.")
            landmarks = np.load(output_path)
        else:
            print("Standardizing histogram based on train split")
            image_paths = train_set._load_paths()[tio.IMAGE]
            landmarks = tio.HistogramStandardization.train(
                image_paths,
                output_path=output_path,
            )
            print('\nTrained landmarks:', landmarks)
        landmarks_dict = {'image': landmarks}
        # histogram_transform = tio.HistogramStandardization(landmarks_dict)

        histogram_transform = tio.Compose([
                                            tio.Blur(std = 0.75),
                                            tio.HistogramStandardization(landmarks=landmarks_dict, masking_method='label'),
                                            ShiftScale(shift = 0, scale = 1 / 2048, random_shift = 0.0, random_scale = 0.0),
                                            # tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                                            # tio.ToCanonical(),
                                            # tio.Resample(1),
                                          ])

    spacings_full = np.array([]).reshape(0, 3)
    classes_full = np.array([])
    histogram_full = []
    histogram_full_standardized = []

    for dataset in [
        train_set,
        val_set,
        test_set]:

        print(f"\nGenerating stats for {dataset.split}")
        spacing_list = np.array([]).reshape(0, 3)
        histogram_list = []
        histogram_list_standardized = []
        classes_list = np.array([])
        sizes_list = []

        img_dir = os.path.join(train_set.root, "histogram_standardized", dataset.split)
        os.makedirs(img_dir, exist_ok=True)

        with tqdm(total=len(dataset), desc=dataset.split, unit='batch', leave=False) as pbar:
            for subject in dataset:

                subject_histogram = get_histogram(subject[tio.IMAGE][tio.DATA]) # get histogram
                subject_spacing = np.array([round(s, 1) for s in subject[tio.IMAGE].spacing]) # get spacing
                subject_classes = np.unique(subject[tio.LABEL][tio.DATA])[1:]  # get classes count, don't get background

                spacing_list = np.vstack((spacing_list, subject_spacing))
                histogram_list.append(subject_histogram)
                classes_list = np.concatenate((classes_list, subject_classes))

                if histogram_standardization:
                    standardized_subject = histogram_transform(subject)
                    standard = get_histogram(standardized_subject[tio.IMAGE][tio.DATA])
                    histogram_list_standardized.append(standard)
                    sizes_list.append(np.asarray(standardized_subject[tio.IMAGE][tio.DATA].shape[-3:]))

                    standardized_subject.plot(show=False, output_path=os.path.join(img_dir, subject['subject_id']))
                    plt.close()

                pbar.update(1)

        plot_histograms(histogram_list=histogram_list, save=dataset.split_dir)
        plot_spacing_histogram(spacings=spacing_list, save=dataset.split_dir)
        plot_label_counts(labels_list=[classes_list], save=dataset.split_dir, remapping = remapping_cervical_thoraic_lumbar)

        if histogram_standardization:
            plot_histograms(histogram_list=histogram_list_standardized, save=f'{dataset.split_dir}_standardized')

        np.save(file=f'{dataset.split_dir}/sizes_{dataset.split}.npy', arr=sizes_list)
        print(dataset.split, sizes_list)
        np.save(file=f'{dataset.split_dir}/histograms_{dataset.split}.npy', arr=histogram_list)
        np.save(file=f'{dataset.split_dir}/histograms_{dataset.split}_standardized.npy', arr=histogram_list_standardized)

        spacings_full = np.vstack((spacings_full, spacing_list))
        histogram_full.extend(histogram_list)
        classes_full = np.concatenate((classes_full, classes_list))

        histogram_full_standardized.extend(histogram_list_standardized)

    plot_histograms(histogram_list=histogram_full, save=train_set.root, show_plot=show)
    plot_spacing_histogram(spacings=spacings_full, save=train_set.root, show_plot=show)
    plot_label_counts(labels_list=[classes_full], save=train_set.root, remapping = remapping_cervical_thoraic_lumbar, show_plot=show)

    if histogram_standardization:
        plot_histograms(histogram_list=histogram_full_standardized, save=f'{train_set.root}_standardized')

    np.save(file=f'{train_set.root}/histograms_verse{str(edition)}.npy', arr=histogram_full)
    np.save(file=f'{train_set.root}/histograms_verse{str(edition)}_standardized.npy', arr=histogram_full_standardized)

if __name__ == "__main__":
    get_verse_dataset_statistics(root='datasets', edition=19,download=True, histogram_standardization=False, show = True)
