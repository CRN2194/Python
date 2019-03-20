import fire
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import join
from skimage import io
from skimage.measure import label, regionprops
from joblib import Parallel, delayed
from collections import namedtuple
import cv2
from skimage.transform import resize
from sklearn.externals import joblib
from skimage.feature import local_binary_pattern as lbp
from skimage.color import rgb2gray
Rect = namedtuple('Rectangle', 'xmin ymin xmax ymax')
clfSVM = joblib.load('/home/cesar/Desktop/testdataset/full/fullSVM6_training.pkl')
clfNN = joblib.load('/home/cesar/Desktop/testdataset/full/fullNN_training.pkl')
class Eval(object):
    """Eval traffic sign dataset """

    def __init__(self):
        pass

    def read_dataset(self, path):
        """ Read dataset images filename
        :param path: Path were the dataset is stored
        :returns: A dictionary with the dataset filenames
        """
        # train
        train_l = glob(join(path, 'train', '*.jpg'))
        dataset = [{'train': t, 'gt': t.replace('train', 'train_masks')} for t in train_l]
        return dataset

    @staticmethod
    def rect_inter_area(a, b):
        """Computes the intersecting area between two rectangles, a and b
        :param a: First rectangle
        :param b: Second rectangle
        :returns: Intersecting area
        """
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        else:
            return 0
    @staticmethod
    def rect_area(a):
        """Computes the area of a rectangle
        :param a: Rectangle
        :returns: Area of the rectangle
        """
        dx = a.xmax - a.xmin
        dy = a.ymax - a.ymin
        if (dx >= 0) and (dy >= 0):
            return dx * dy

    @staticmethod
    def segment_image(im, color_space):
        """Segments an image using a given color space
        :param im: Input image (x,x,3)
        :param color_space: Color space required for segmentation
        :returns: uint8 mask. 255 true, 0 False
        """
        ## Add Segmentation code HERE
        if color_space == 'HSV':
            im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            # Then the color channels to be used for segmentation in HSV are separated
            # red1
            red1_low = np.array([0, 217, 83], dtype=np.uint8)
            red1_high = np.array([10, 255, 255], dtype=np.uint8)
            # red2
            red2_low = np.array([155, 120, 50], dtype=np.uint8)
            red2_high = np.array([179, 240, 235], dtype=np.uint8)
            # red4
            red4_low = np.array([150, 142, 102], dtype=np.uint8)
            red4_high = np.array([179, 204, 255], dtype=np.uint8)
            # red5 rose
            red5_low = np.array([160, 43, 58], dtype=np.uint8)
            red5_high = np.array([170, 196, 143], dtype=np.uint8)
            # blue
            blue1_low = np.array([115, 132, 160], dtype=np.uint8)
            blue1_high = np.array([130, 255, 255], dtype=np.uint8)
            # morado
            mo1_low = np.array([125, 76, 76], dtype=np.uint8)
            mo1_high = np.array([143, 255, 255], dtype=np.uint8)
            # yellow black
            yellow_lows = np.array([16, 226, 90], dtype=np.uint8)
            yellow_high = np.array([22, 255, 180], dtype=np.uint8)

            # Color ranges used for the specific channel
            r1mask = cv2.inRange(im, red1_low, red1_high)
            r2mask = cv2.inRange(im, red2_low, red2_high)
            r4mask = cv2.inRange(im, red4_low, red4_high)
            r5mask = cv2.inRange(im, red5_low, red5_high)
            bmask = cv2.inRange(im, blue1_low, blue1_high)
            m1mask = cv2.inRange(im, mo1_low, mo1_high)
            ymask = cv2.inRange(im, yellow_lows, yellow_high)

            # Creation of the mask, which will add all previously used ranges
            mask = cv2.add(r1mask, r2mask)
            mask = cv2.add(mask, r4mask)
            mask = cv2.add(mask, r5mask)
            mask = cv2.add(mask, bmask)
            mask = cv2.add(mask, m1mask)
            mask = cv2.add(mask, ymask)

            # Use of morphology for the improvement of results obtained with pure segmentation
            kerneld = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kerneld)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kerneld)



            mask = Eval.search_blobs(closing)
            # The rec_blol function is called which will return a mask with the connected components.
            # for the time of execution is commented on the line below, is implemented but not used in all images for the time that this takes to run
            # mask=Eval.rec_blob(mask)
            mask = cv2.dilate(mask, kerneld, iterations=9)
            mask = cv2.erode(mask, kerneld, iterations=3)

            # Then the change of the mask will be made to calculate the metrics, so a conversion of 255 to 1 will be made
            blanco = (mask[:] == 255)
            mask[blanco] = [1]
            negro = (mask[:] == 0)
            mask[negro] = [0]
        return mask
        ## Code ends

    @staticmethod
    def search_blobs(mask):
        maskb = np.uint8(mask)


        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(maskb, connectivity=4)

        sizes = stats[1:, -1];
        nb_components = nb_components - 1
        minsize = 1100
        maxsize = 3500
        blobs = np.zeros((output.shape))

        for i in range(0, nb_components):
            if sizes[i] >= minsize and sizes[i]<=maxsize :
                blobs[output == i + 1] = 255

        return  blobs
    @staticmethod
    def eval_mask(gt, mask):
        """ Eval mask
        :param gt: Groundtruth mask (binary values)
        :param mask: Obtained mask (binary values)
        :returns: tp, tn, fp, fn in pixels
        """
        tp_m = (gt == 1) & (mask == 1)
        tn_m = (gt == 0) & (mask == 0)
        fp_m = (gt == 0) & (mask == 1)
        fn_m = (gt == 1) & (mask == 0)
        return (tp_m.sum(), tn_m.sum(), fp_m.sum(), fn_m.sum())

    @staticmethod
    def eval_recognition(bb_sol, bb_mask):
        """ Eval mask
        :param bb_sol: List of Rect bounding boxes obtained after classification
        :param bb_mask: List of Rect bounding boxes from the GT mask
        :returns: tp, tn, fp, fn in windows
        """
        used_mask = []
        tp = 0
        for bb_s in bb_sol:
            for i, bb_m in enumerate(bb_mask):
                if i not in used_mask:
                    if Eval.rect_inter_area(bb_s, bb_m) > 0.5 * Eval.rect_area(bb_m) and Eval.rect_area(bb_s) < 1.5 * Eval.rect_area(bb_m):
                        tp += 1
                        used_mask.append(i)
        fn = len(bb_mask) - tp
        fp = len(bb_sol) - tp
        return (tp, fp, fn)

    @staticmethod
    def classify(im, mask,cla="SVM"):
        """ Classification of blobs
        :im: Image with traffic signs
        :mask: Mask after segmentation
        :returns: List of bounding boxes
        """
        # Training

        bb_sol = []
        label_im = label(mask >200)
        props = regionprops(label_im)
        if cla=="SVM":
            for p in props:

                coordinates = Rect(xmin=p.bbox[0], ymin=p.bbox[1], xmax=p.bbox[2], ymax=p.bbox[3])
                signs = im[coordinates.xmin:coordinates.xmax, coordinates.ymin:coordinates.ymax, :]
                signs_found = resize(signs, (32, 32))
                io.imsave("/home/cesar/Desktop/datasetminin/gg.jpg", signs)
                datasetlbp = lbp(rgb2gray(signs_found), 8, 3, method='uniform')
                hist = np.histogram(datasetlbp, normed=True, bins=8, range=(0, 8))
                input_predict = hist[0]
                predict = clfSVM.predict([input_predict])
                if predict[0] == 1:
                    bb_sol.append(coordinates)


            return bb_sol
        if cla=="NN":
            for p in props:

                coordinates = Rect(xmin=p.bbox[0], ymin=p.bbox[1], xmax=p.bbox[2], ymax=p.bbox[3])
                signs = im[coordinates.xmin:coordinates.xmax, coordinates.ymin:coordinates.ymax, :]
                signs_found = resize(signs, (32, 32))
                datasetlbp = lbp(rgb2gray(signs_found), 8, 3, method='uniform')
                hist = np.histogram(datasetlbp, normed=True, bins=8, range=(0, 8))
                input_predict = hist[0]
                predict = clfNN.predict([input_predict])
                if predict[0] == 1:
                    bb_sol.append(coordinates)

            return bb_sol

    @staticmethod
    def process_image(ip, color_space):
        """ Process one image
        :param ip: name of train and gt image files
        :returns: (tp,tn,fp,fn)
        """
        # Segment image

        im = io.imread(ip['train'])
        gt = io.imread(ip['gt'])
        mask = Eval.segment_image(im, color_space)

        # Eval Segmentation
        (tp, tn, fp, fn) = Eval.eval_mask((gt > 200), mask)

        # Classify blobs
        bb_sol = Eval.classify(im, mask)

        # Get bounding boxes from GT
        bb_mask = []
        label_im = label(gt > 200)
        props = regionprops(label_im)
        for p in props:
            props = Rect(xmin=p.bbox[0], ymin=p.bbox[1], xmax=p.bbox[2], ymax=p.bbox[3])
            bb_mask.append(props)

        # Eval Traffic sign recognition
        (tp_r, fp_r, fn_r) = Eval.eval_recognition(bb_sol, bb_mask)
        return (tp, tn, fp, fn, tp_r, fp_r, fn_r)

    def process_dataset(self, dataset, color_space, njobs):
        """ Process the full dataset
        :param dataset: Dicctionary that contains all dataset filenames
        :param color_space: Color space required for segmentation
        :param njobs: Number of cores
        :returns: None
        """
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        TP_r = 0
        FP_r = 0
        FN_r = 0

        res = Parallel(n_jobs=njobs, verbose=4)(delayed(Eval.process_image)(ip, color_space) for ip in dataset)
        for r in res:
            TP += r[0]
            TN += r[1]
            FP += r[2]
            FN += r[3]
            TP_r += r[4]
            FP_r += r[5]
            FN_r += r[6]

        # Segmentation
        precision = TP / (TP + FP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        recall = TP / (TP + FN)
        print('\nSegmentation\n')
        print('Precision = ', precision)
        print('Accuracy = ', accuracy)
        print('Recall = ', recall)
        print('F-measure = ', 2 * (precision * recall) / (precision + recall))

        # Recognition
        if TP_r + FP_r == 0:
            print('\nRecognition\n')
            print('Precision = 0')
            print('Accuracy = 0')
            print('Recall = 0')

        else:
            precision = TP_r / (TP_r + FP_r)
            accuracy = TP_r / (TP_r + FN_r)
            recall = TP_r / (TP_r + FN_r + FP_r)
            print('\nRecognition\n')
            print('Precision = ', precision)
            print('Accuracy = ', accuracy)
            print('Recall = ', recall)

    def eval(self, path, njobs=1, color_space='HSV'):
        """ Segmentation evaluation on training dataset
        :param path: Dataset path
        :param color_space: Select segmentation color space
        :returns: None
        """
        dataset = self.read_dataset(path)
        self.process_dataset(dataset, color_space, njobs)


if __name__ == "__main__":
    fire.Fire(Eval)