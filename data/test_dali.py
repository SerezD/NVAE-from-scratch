from matplotlib import gridspec
from matplotlib import pyplot as plt
from nvidia.dali import pipeline_def, fn


@pipeline_def
def image_label_pipeline():

    # initial fill = dimension of the shuffling buffer
    jpegs, labels = fn.readers.file(file_root=image_dir, random_shuffle=True, initial_fill=4)
    images = fn.decoders.image(jpegs, device='cpu')
    images = fn.resize(images.gpu(), size=256)
    return images, labels


def show_images(image_batch):
    columns, rows = 2, 2
    fig = plt.figure(figsize=(24, 24))
    gs = gridspec.GridSpec(rows, columns)

    for j in range(rows * columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch.at(j))
    plt.show()
    plt.close(fig)


if __name__ == '__main__':

    image_dir = './samples/with_classes/'
    max_batch_size = 4

    pipe = image_label_pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)
    pipe.build()

    images, labels = pipe.run()
    show_images(images.as_cpu())
