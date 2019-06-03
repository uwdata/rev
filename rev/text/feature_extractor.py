import pandas as pd
import numpy as np
from collections import defaultdict


filtered_columns = [
            'fig_id', 'fig_fn',
            'fw', 'fh',
            'vscore', 'hscore',
            'vrange', 'hrange',
            'vfreq', 'hfreq',
            'x', 'y',
            'x2', 'y2',
            'xc', 'yc',
            'w', 'h',
            'xp', 'yp',
            'x2p', 'y2p',
            'xcp', 'ycp',
            'wp', 'hp',
            'aspect',
            'ang',
            'quad',
            'bw', 'bh',
            'bwp', 'bhp',
            'u', 'v',
            'u2', 'v2',
            'uc', 'vc',
            'du', 'dv',
            'adu', 'adv',
            'rad',
            'text',
            'type'
        ]


def from_chart(chart):
    """
    Extract geometric features from bounding boxes in a chart.
    Assuming all the boxes belong to a figure.

    :param chart: chart object
    :return: panda DataFrame with features for all the boxes.
    """
    text_boxes = chart.text_boxes
    fh, fw, _ = chart.image.shape
    features = from_text_boxes(text_boxes, (fh, fw), chart.id, chart.filename)

    return features


def from_text_boxes(boxes, (fh, fw), chart_id, chart_fn=''):
    """
    Extract geometric features from bounding boxes.
    Assuming all the boxes belong to a figure.

    :param boxes:
    :param (fh, fw):
    :param chart_id: figure id.
    :param chart_fn: file name for debug.
    :return: panda DataFrame with features for all the boxes
    """
    rows = []
    for box in boxes:
        vscore, hscore, vrange, hrange, vfreq, hfreq = alignment_scores(box, boxes, (fh, fw))

        row = box.to_dict()
        row['vscore'] = vscore
        row['hscore'] = hscore
        row['vrange'] = vrange
        row['hrange'] = hrange
        row['vfreq'] = vfreq
        row['hfreq'] = hfreq
        # TODO(jpocom)
        # temporal fix, because in this class we use 'w' and 'h' instead of 'width' and 'height'
        row['w'] = row['width']
        row['h'] = row['height']
        row['fig_id'] = chart_id
        row['fig_fn'] = chart_fn
        row['fw'] = fw
        row['fh'] = fh

        rows.append(row)

    df = pd.DataFrame(rows)

    if not rows:
        return df

    # right-bottom coordinate
    df['x2'] = df.x + df.w
    df['y2'] = df.y + df.h
    # center coordinate
    df['xc'] = df.x + df.w / 2.0
    df['yc'] = df.y + df.h / 2.0

    # normalized top-left coordinate
    df['xp'] = df.x / fw
    df['yp'] = df.y / fh
    # normalized right-bottom coordinate
    df['x2p'] = df.x2 / fw
    df['y2p'] = df.y2 / fh
    # normalized center coordinate
    df['xcp'] = df.xc / fw
    df['ycp'] = df.yc / fh
    # normalized box size
    df['wp'] = df.w / fw
    df['hp'] = df.h / fh

    # aspect ratio in log-10 units
    df['aspect'] = np.log10(df.w / df.h)

    # angle from actual center [-1,1]
    # 0+ -> counter-clockwise from positive x-axis
    df['ang'] = np.arctan2(df.yc - fh / 2.0, df.xc - fw / 2.0) / np.pi
    # discretize angles into quadrants (0, 1, 2, 3)
    df['quad'] = np.floor(2 * (df.ang + 1)) % 4

    xmin = df['x'].min()
    ymin = df['y'].min()
    x2max = df['x2'].max()
    y2max = df['y2'].max()

    # bounding-width (bw) and bounding-height (bh)
    # bounding box of region containing text boxes
    df['bw'] = (x2max - xmin)
    df['bh'] = (y2max - ymin)
    df['bwp'] = df.bw / fw
    df['bhp'] = df.bh / fh

    # normalized top-left coordinate in container box
    df['u'] = (df.x - xmin) / df.bw
    df['v'] = (df.y - ymin) / df.bh

    # normalized bottom-right coordinate in container box
    df['u2'] = (df.x2 - xmin) / df.bw
    df['v2'] = (df.y2 - ymin) / df.bh

    # normalized bottom-right coordinate in container box
    df['uc'] = (df.xc - xmin) / df.bw
    df['vc'] = (df.yc - ymin) / df.bh

    def extremum(a, b):
        return np.where(abs(b) < abs(a), a, b)

    df['du'] = extremum(2 * df.u - 1, 2 * df.u2 - 1)
    df['dv'] = extremum(2 * df.v - 1, 2 * df.v2 - 1)

    # absolute extremal point [0,1]
    df['adu'] = abs(df.du)
    df['adv'] = abs(df.dv)

    # radius from normalized center
    df['rad'] = np.sqrt(df.du * df.du + df.dv * df.dv)

    return df[filtered_columns]


def alignment_scores(ref_box, boxes, (fh, fw)):
    """
    Return the number of boxes which intersect vertically and horizontally the
    'ref_box'. These values are normalized by the total number of boxes.

    :param ref_box:  reference box
    :param boxes: set of boxes in figure
    :return: (vscore, hscore, vrange, hrange, vfreq, hfreq)
    """
    getters = {
        'vert': {'left': lambda b: b.x1, 'right': lambda b: b.x2, 'center': lambda b: b.xc},
        'hori': {'left': lambda b: b.y1, 'right': lambda b: b.y2, 'center': lambda b: b.yc}
    }

    r = ref_box
    count = defaultdict(int)
    aboxes = defaultdict(list)
    span = defaultdict(float)
    freq = defaultdict(float)
    th = 3
    for orient in ['vert', 'hori']:
        getter = getters[orient]

        for b in boxes:
            aligned = {'left': True, 'right': True, 'center': True}
            for pos in aligned.keys():
                if abs(getter[pos](r) - getter[pos](b)) > th:
                    aligned[pos] = False

            if any(aligned.values()):
                count[orient] += 1
                aboxes[orient].append(b)

        getter = getters['vert' if orient == 'hori' else 'hori']
        aboxes[orient].sort(key=lambda b: getter['center'](b))

        values = [getter['center'](b) for b in aboxes[orient]]
        span[orient] = abs(values[-1] - values[0])

        pos = np.searchsorted(values, getter['center'](r))
        values[pos] += 1e10
        closest = np.argmin(np.abs(np.array(values) - getter['center'](r)))
        values[pos] -= 1e10

        freq[orient] = 0.0 if span[orient] == 0 else 1.0 - abs(values[closest] - values[pos]) / span[orient]

    num_boxes = float(len(boxes))
    return count['vert'] / num_boxes, count['hori'] / num_boxes, \
           span['vert'] / fh, span['hori'] / fw, \
           freq['vert'], freq['hori']
