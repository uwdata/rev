import numpy as np


def rgba2rgb(img):
    """
    Convert the rgba image into a rgb with white background.
    :param img:
    :return:
    """
    arr = img.astype('float') / 255.
    alpha = arr[..., -1]
    channels = arr[..., :-1]
    out = np.empty_like(channels)

    background = (1, 1, 1)
    for ichan in range(channels.shape[-1]):
        out[..., ichan] = np.clip(
            (1 - alpha) * background[ichan] + alpha * channels[..., ichan],
            a_min=0, a_max=1)

    return (out * 255.0).astype('uint8')


def ttoi(t):
    """
    Converts tuples values to tuple of rounded integers.
    """
    return tuple(map(int, map(round, t)))


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels:
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print


def create_predicted1_bbs(chart):
    """
    Create an empty bbs file with empty texts and types.
    :param chart:
    :return:
    """
    import rev
    ifn = chart.filename.replace('.png', '-texts.csv')
    text_boxes = rev.chart.load_texts(ifn)

    # cleaning type field
    for text_box in text_boxes:
        text_box._type = ''

    ofn = chart.filename.replace('.png', '-pred1-texts.csv')
    rev.chart.save_texts(text_boxes, ofn)
