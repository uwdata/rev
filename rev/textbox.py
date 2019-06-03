import numpy as np
from collections import OrderedDict
import cv2


class TextBox(object):
    def __init__(self, id, x, y, w, h, type='', text=''):
        self._id = id
        self._rect = (float(x), float(y), float(w), float(h))
        self._type = type
        self._text = text

        self._regions = []  # the connected components

    def __str__(self):
        return 'bbox[{4}]: [{0} {1} {2} {3}] : [{5}] : [{6}]'\
            .format(self.x, self.y, self.w, self.h, self._id, self._type, self._text)

    __repr__ = __str__

    def center(self):
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    def area(self):
        return self.w * self.h

    # def expand(self, factor):
    #     _w, _h = self.w * factor, self.h * factor
    #     _x = self.x - (_w - self.w) / 2.0
    #     _y = self.y - (_h - self.h) / 2.0
    #     return TextBox(self._id, self._type, _x, _y, _w, _h)

    def wrap_rect(self, (fh, fw), padx=2, pady=None):
        pady = padx if pady is None else pady
        nx, ny = max(self.x - padx, 0), max(self.y - pady, 0)
        nw = min(self.x + self.w + padx, fw) - nx
        nh = min(self.y + self.h + pady, fh) - ny
        self._rect = (nx, ny, nw, nh)

    @property
    def type(self):
        return self._type

    @property
    def text(self):
        return self._text

    @property
    def x(self):
        return self._rect[0]

    @property
    def y(self):
        return self._rect[1]

    @property
    def w(self):
        return self._rect[2]

    @property
    def h(self):
        return self._rect[3]

    @property
    def rect0(self):
        x, y, w, h = self._rect
        return [x, y, x + w - 1, y + h - 1]

    @property
    def x1(self):
        return self._rect[0]

    @property
    def y1(self):
        return self._rect[1]

    @property
    def x2(self):
        return self.x + self.w - 1

    @property
    def y2(self):
        return self.y + self.h - 1

    @property
    def xc(self):
        return self.x + self.w / 2.0

    @property
    def yc(self):
        return self.y + self.h / 2.0

    @property
    def p1(self):
        return self.x, self.y

    @property
    def p2(self):
        return self.x + self.w - 1, self.y + self.h - 1

    def to_dict(self):
        row = OrderedDict()
        row['id'] = self._id
        row['x'] = self.x
        row['y'] = self.y
        row['width'] = self.w
        row['height'] = self.h
        row['text'] = self._text
        row['type'] = self._type
        return row

    def jaccard_similarity(self, tbox):
        """
        Calculates the Jaccard similarity (the similarity used in the
        PASCAL VOC)
        Note: the are could be computed as:
           area_intersection = bbox.copy().intersect(self).area()
              but we replicate the code for efficency reason.

        copied from https://github.com/lorisbaz/self-taught_localization/blob/master/textbox.py
        """
        xmin = max(self.x, tbox.x)
        ymin = max(self.y, tbox.y)
        xmax = min(self.x + self.w, tbox.x + tbox.w)
        ymax = min(self.y + self.h, tbox.y + tbox.h)
        if (xmin > xmax) or (ymin > ymax):
            xmin = 0.0
            ymin = 0.0
            xmax = 0.0
            ymax = 0.0

        area_intersection = np.abs(xmax - xmin) * np.abs(ymax - ymin)
        area_union = self.area() + tbox.area() - area_intersection
        return area_intersection / float(area_union)

    def matching_score(self, tbox):
        """
        Score use to determine the matching between two rectangles.
        http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=D7BFE2DA34118919E31A9A2FC5F85170?doi=10.1.1.104.1667&rep=rep1&type=pdf

        Note matching_score >= jaccard_similarity
        :param tbox:
        :return:
        """
        xmin = max(self.x, tbox.x)
        ymin = max(self.y, tbox.y)
        xmax = min(self.x + self.w, tbox.x + tbox.w)
        ymax = min(self.y + self.h, tbox.y + tbox.h)
        if (xmin > xmax) or (ymin > ymax):
            xmin = 0.0
            ymin = 0.0
            xmax = 0.0
            ymax = 0.0

        area_intersection = np.abs(xmax - xmin) * np.abs(ymax - ymin)
        if self.area() + tbox.area() == 0:
            return 0

        return 2.0 * area_intersection / (self.area() + tbox.area())

    def find_best_match(self, texts, return_all=False):
        coeffs = []
        for tbox in texts:
            coeff = self.matching_score(tbox)
            coeffs.append((tbox, coeff))

        if return_all:
            return coeffs

        return max(coeffs, key=lambda t: t[1])

    @staticmethod
    def merge_boxes(texts, id=0):
        points = []
        new_tbox = TextBox(id, 0, 0, 0, 0)
        for tbox in texts:
            points.append(ru.points(tbox.rect))
            new_tbox._regions.extend(tbox.regions)

        new_tbox._rect = cv2.boundingRect(np.concatenate(points).astype('float32'))

        return new_tbox

    @property
    def num_comp(self):
        return len(self._regions)

