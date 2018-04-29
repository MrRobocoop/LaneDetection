import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelextrema as ag
#from goto import with_goto

class line_class(object):
    def __init__(self, base):
        self.line_type = 'unknown'
        self.points = np.uint32([[]])
        self.point_base = base
        self.points = np.append(self.points, np.uint32([base]),axis = 1)
        self.line_parameter = []
        self.slice = np.uint32([[base]])
        self.first_slice = base[1]

    def measure_distance(self, point):
        distance = np.sqrt(np.sum(np.square(self.point_base - point)))
        return distance

    def add_new_point(self, point):

        self.points = np.append(self.points, np.uint32([point]), axis=0)
        self.point_base = point

    def get_slice(self):
        last_slice = self.first_slice
        for point in self.points:
            if point[1] != last_slice:
                #add a new slice
                np.append(self.slice, np.array([[point]]),axis = 0)
                last_slice = point[1]
            else:
                #add a new point in slice
                np.append(self.slice, np.array([[point]]), axis=1)
        return self.slice

def classify_line(point, list_line, margin):
    create_line = locals()
    if len(list_line) > 0:
        new_line_counter = 0;
        for i in range(0,len(list_line)):
            if(list_line[i].measure_distance(point)<margin):
                list_line[i].add_new_point(point)
                break
            else:
                new_line_counter = new_line_counter + 1

        if (new_line_counter == len(list_line)):
            pass
            #line_id = str(len(list_line) + 1)
            #create_line['line' + line_id] = line_class(point)
            #exec ("list_line.append(line" + line_id + ")")

    else:
        line_id = str(len(list_line)+1)
        create_line['line'+line_id] = line_class(point)
        exec("list_line.append(line"+line_id+")")

    return list_line

def find_lines(img, margin):
    right_side_x = []
    right_side_y = []
    left_side_x = []
    left_side_y = []
    list_line = []
    past_cord = 0
    past_y_cord = 0
    # right side
    for i in reversed(range(1, 100,2)):
        histogram = np.sum(img[i * img.shape[0] / 100:(i + 1) * img.shape[0] / 100, 0: img.shape[1]], axis=0)

        histogram = gaussian_filter1d(histogram, 2, order=0)
        hist_nonzero = np.nonzero(histogram)
        #print hist_nonzero
        #print hist_nonzero[0].size
        if hist_nonzero[0].size > 0:
            xcordx = ag(histogram[hist_nonzero[0]], np.greater_equal, order=5)
            #print xcordx
        #xcordx = np.where(histogram > 512)
            xcordx = hist_nonzero[0][xcordx[0]]
            #print xcordx
            ycord = int(i* img.shape[0] / 100)
        #print xcordx

            if len(xcordx) > 0:
                for xcord in xcordx:
                    list_line = classify_line(np.array([xcord, ycord]), list_line, margin)

    idx = 0
    #print len(list_line)
    for line_this in list_line:
        if np.size(line_this.points)<5:
            del list_line[idx]
            idx = idx-1
        idx = idx+1
    #print len(list_line)
    return list_line