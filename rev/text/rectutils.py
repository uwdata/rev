"""
Based on https://github.com/szakrewsky/text-search/blob/master/rectutils.py
"""
import cv2
import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from shapely.geometry import LineString

from .. import utils as u

def on_same_line(r1, r2, horiz=True):
    x1, y1, w1, h1 = r1 if horiz else (r1[1], r1[0], r1[3], r1[2])
    x2, y2, w2, h2 = r2 if horiz else (r2[1], r2[0], r2[3], r2[2])

    over, d = range_overlap((y1, y1+h1), (y2, y2+h2))
    if over and d > min(h1, h2) / 2.0:
        return True
    return False


def next_on_same_line(r1, r2, dist=None, horiz=True):
    x1, y1, w1, h1 = r1 if horiz else (r1[1], r1[0], r1[3], r1[2])
    x2, y2, w2, h2 = r2 if horiz else (r2[1], r2[0], r2[3], r2[2])
    dist = min(h1, h2) / float(2) if dist is None else dist

    if not on_same_line(r1, r2, horiz=horiz) or abs(x1 + w1 - x2) > dist:
        return False
    return True


def on_consecutive_line(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if abs(y1 + h1 - y2) > min(h1,h2)/float(2):
        return False
    return True


def same_height(r1, r2, max_diff=None, horiz=True):
    x1, y1, w1, h1 = r1 if horiz else (r1[1], r1[0], r1[3], r1[2])
    x2, y2, w2, h2 = r2 if horiz else (r2[1], r2[0], r2[3], r2[2])
    max_diff = min(h1, h2) if max_diff is None else max_diff

    if abs(h1 - h2) > max_diff:
        return False
    return True


def overlap(r1, r2):
    """
    Based on http://codereview.stackexchange.com/questions/31352/overlapping-rectangles
    Overlapping rectangles overlap both horizontally & vertically
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    over1, _ = range_overlap((x1, x1 + w1), (x2, x2 + w2))
    over2, _ = range_overlap((y1, y1 + h1), (y2, y2 + h2))
    return over1 and over2


def range_overlap((a_min, a_max), (b_min, b_max)):
    """
    Based on http://codereview.stackexchange.com/questions/31352/overlapping-rectangles
    Neither range is completely greater than the other
    """
    if (a_min <= b_max) and (b_min <= a_max):
        return True, min(a_max, b_max) - max(a_min, b_min) + 1
    return False, -1


def inside(r1, r2):
    """
    Check if r1 is inside r2
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return (x1 >= x2) and (y1 >= y2) and (x1+w1 <= x2+w2) and (y1+h1 <= y2 + h2)


def rect_segment_intersection(rect, seg):
    """
    Returns the intersection of a rectangle rect and seg
    :param rect: (x, y, w, h)
    :param seg: (point1, point2)
    :return tuple(x, y, v), where
    (x, y) is the intersection
    v == False if there are 0 or inf. intersections (invalid)
    v == True  if it has a unique intersection ON the segment
    """
    x, y, w, h = rect
    segments = LineString([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
    segment = LineString(seg)
    inter = segments.intersection(segment)
    if inter.geom_type == 'Point':
        return inter.x, inter.y, True

    return 0, 0, False


def center(r):
    x, y, w, h = r
    w = w if w > 2 else w - 0.1
    h = h if h > 2 else h - 0.1
    return x + w / 2.0, y + h / 2.0

def filter_duplicates(rects):
    print "Filtering %d regions..." % (len(rects))

    th = 10
    C = np.zeros((len(rects), len(rects)), dtype=bool)
    for i, r1 in enumerate(rects):
        for j, r2 in enumerate(rects):
            # if abs(r1[0] - r2[0]) < th and abs(r1[1] - r2[1]) < th and \
            #    abs(r1[2] - r2[2]) < th and abs(r1[3] - r2[3]) < th:
            #     C[i, j] = True
            # if overlap(r1, r2):
            #     C[i, j] = True
            if inside(r1, r2):
                C[i, j] = True

    rects, group_indices = __bfs_bbx(rects, C)

    print "\tto %d regions" % (len(rects))
    return rects, group_indices


def mean_color(img, bw, rect):
    x, y, w, h = wrap_rect(rect, bw.shape, padx=1)
    roi = img[y:y + h, x:x + w, :]
    roi_bw = bw[y:y + h, x:x + w]

    pos = np.transpose(np.nonzero(roi_bw))
    rows = pos[:, 0]
    cols = pos[:, 1]
    meancolor = np.mean(roi[rows, cols], axis=0)

    # vis = cv2.cvtColor(roi_bw, cv2.COLOR_GRAY2BGR)
    # vis[roi_bw==255] = meancolor
    #
    # vis = np.hstack((roi, cv2.cvtColor(roi_bw, cv2.COLOR_GRAY2BGR), vis))
    # show_image('img (%d, %d, %d, %d)' % (x, y, w, h), vis)

    return meancolor


def color_dist(img, bw, r1, r2):
    c1 = mean_color(img, bw, r1)/255.
    c2 = mean_color(img, bw, r2)/255.

    c1_lab = convert_color(sRGBColor(c1[2], c1[1], c1[0]), LabColor)
    c2_lab = convert_color(sRGBColor(c2[2], c2[1], c2[0]), LabColor)
    delta_e = delta_e_cie2000(c1_lab, c2_lab)
    return delta_e


def find_words(rects, img):
    C = np.zeros((len(rects), len(rects)), dtype=bool)
    for i, r1 in enumerate(rects):
        x1, y1, w1, h1 = r1
        for j, r2 in enumerate(rects):
            x2, y2, w2, h2, = r2
            if i == j or \
               (abs(y1 - y2) < min(h1, h2)/float(2) and                 # almost same level
                abs(h1 - h2) < min(h1, h2)/2. and                          # almost same height
                # (inside(r1, r2) or inside(r2, r1)) and
                # (abs(x1 + w1 - x2) < 10 or abs(x2 + w2 - x1) < 10) and  # boxes distance
                (abs(x1 + w1 - x2) < 10 or abs(x2 + w2 - x1) < 10 or inside(r1, r2) or inside(r2, r1)) and  # boxes distance
                color_dist(img, r1, r2) < 10):                          # almost same color
                C[i, j] = True

    rects, group_indices = __bfs_bbx(rects, C)
    return rects


def find_words2(rects, img):
    C = np.zeros((len(rects), len(rects)), dtype=bool)
    for i, r1 in enumerate(rects):
        y1, x1, h1, w1 = r1
        for j, r2 in enumerate(rects):
            y2, x2, h2, w2, = r2
            if i == j or \
               (abs(y1 - y2) < min(h1, h2)/float(2) and                 # almost same level
                abs(h1 - h2) < min(h1, h2)/float(2) and                 # almost same height
                (abs(x1 + w1 - x2) < 10 or abs(x2 + w2 - x1) < 10) and  # boxes distance
                color_dist(img, r1, r2) < 10):                          # almost same color
                C[i, j] = True

    rects, group_indices = __bfs_bbx(rects, C)
    return rects


def __bfs_bbx(rects, C):
    visited = set()
    isclose = {}
    for i in range(0, len(rects)):
        if i in visited:
            continue

        visited.add(i)
        neighbors = isclose.get(i, [])
        neighbors.extend(np.where(C[i] == True)[0])
        isclose[i] = neighbors
        visited = visited | set(neighbors)

        j = 0
        while j < len(neighbors):
            s = neighbors[j]
            s_neighbors = set(np.where(C[s] == True)[0])
            s_neighbors = s_neighbors - visited
            neighbors.extend(s_neighbors)
            visited = visited | s_neighbors
            j += 1

    newrects = []
    group_indices = []
    for value in isclose.values():
        newrects.append(cv2.boundingRect(np.concatenate([u.points(rects[r]) for r in
                                                         value])))
        group_indices.append(value)

    return newrects, group_indices


def points(rect):
    x, y, w, h = rect
    return [(x, y), (x+w-1, y), (x+w-1, y+h-1), (x, y+h-1)]


# def wrap_rect(rect, (fh, fw), padx=2, pady=None):
#     if pady is None:
#         pady = padx
#     x, y, w, h = rect
#     nx, ny = max(x - padx, 0), max(y - pady, 0)
#     nw = min(x + w + padx, fw) - nx
#     nh = min(y + h + pady, fh) - ny
#     return nx, ny, nw, nh
