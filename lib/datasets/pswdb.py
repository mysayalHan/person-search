#encoding:utf-8
"""
*************************************************************************
    > File Name: pswdb.py
    > Author: CharlesHan
    > Mail: mysayalhan@gmail.com 
    > Created Time: 2018年07月29日 星期日 12时20分57秒
************************************************************************
"""

import json
import os
import os.path as osp 

import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, precision_recall_curve

import datasets
from datasets.imdb import imdb 
from fast_rcnn.config import cfg 
import cPickle

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union 

class pswdb(imdb):
    def __init__(self, image_set, root_dir=None):
        super(pswdb, self).__init__('pswdb_' + image_set)
        self._image_set = image_set
        self._root_dir = self._get_default_path() if root_dir is None \
                         else root_dir
        self._data_path = osp.join(self._root_dir, 'frames')
        self._classes = ('__background__', 'person')
        self._image_index = self._load_image_set_index()
        self._probes = self._load_probes()
        self._roidb_handler = self.gt_roidb
        self._image_ext = '.jpg'
        self._id_train = self._get_trainid_dict()
        assert osp.isdir(self._root_dir), \
                "PSWDB does not exist: {}".format(self._root_dir)
        assert osp.isdir(self._data_path), \
                "Path does not exist: {}".format(self._data_path)
    
    def _get_trainid_dict(self):
        txt_file = osp.join(self._root_dir, 'trainID.txt')
        print txt_file
        trainid_dict = {}
        if osp.isfile(txt_file):
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip('\n')
                    key, value = line.split(' ')
                    trainid_dict[int(key)] = value
        return trainid_dict 

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        image_path = osp.join(self._data_path, index+self._image_ext)
        assert osp.isfile(image_path), \
                "Path does not exist: {}".format(image_path)
        return image_path

    def gt_roidb(self):
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.isfile(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_mat_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb
    
    def _load_mat_annotation(self, index):
        """
        Load image and bounding boxes info from .mat file in the dataset PRW
        format.
        """
        anno_file = osp.join(self._root_dir,'annotations/',index+'.jpg.mat')
        anno = loadmat(anno_file)
        for key in anno.iterkeys():
            # There are different keys in PRW dataset, including
            # 'anno_previous', 'anno_file' and 'box_new'
            if cmp(key,'__header__') == 0:
                continue
            if cmp(key, '__globals__') == 0:
                continue
            if cmp(key, '__version__') == 0:
                continue 

            num_pers = len(anno[key])
            boxes = np.zeros((num_pers, 4), dtype = np.int32)
            gt_classes = np.ones((num_pers), dtype = np.int32)
            overlaps = np.zeros((num_pers, self.num_classes), dtype = np.float32)
            pids = np.zeros((num_pers), dtype = np.int32)
            
            pers = anno[key]
            for ix, per in enumerate(pers):
                boxes[ix, :] = per[1:]
                if int(per[0]) == 932:
                    print 'Fuckkkkkkkkkkkkkkkkkkkkkkkkkkk'
                # For convenience, we use -1 to represent unlabeled perdestrian
                # which is the same as the dataset CUHK
                if int(per[0]) == -2:
                    pids[ix] = -1
                else:
                    pids[ix] = self._id_train[int(per[0])]
                    if pids[ix] > 500:
                        print 'Fuckkkkkkkkkkkkkkkkkkkkkkkkkkk'
                overlaps[:, 1] = 1.0
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            overlaps = csr_matrix(overlaps)
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'gt_pids': pids,
                'flipped': False}
    
    def _is_contain_gt(self, label, index):
        """
        Identify contains corresponding label 
        """
        anno_file = osp.join(self._root_dir, 'annotations/',index+'.jpg.mat')
        anno = loadmat(anno_file)
        flag = 0
        for key in anno.iterkeys():
            # There are different keys in PRW dataset, including
            # 'anno_previous', 'anno_file' and 'box_new'
            if cmp(key,'__header__') == 0:
                continue
            if cmp(key, '__globals__') == 0:
                continue
            if cmp(key, '__version__') == 0:
                continue

            pers = anno[key]
            for ix, per in enumerate(pers):
                if int(per[0]) == int(label):
                    flag += 1
        assert flag < 2
        return flag

    def _get_default_path(self):
        return osp.join(cfg.DATA_DIR, 'pswdb', 'PRW-v16.04.20')

    def _load_image_set_index(self):
        """
        Load the training indexes listed in the dataset's image set file.
        """     
        if self._image_set == 'train':
            train_data = loadmat(osp.join(self._root_dir, 'frame_train.mat'))
            train_data = train_data['img_index_train'].squeeze()
            train = []
            for index, item in enumerate(train_data):
                train.append(str(item[0]))
            return train
        else:
            assert self._image_set == 'test'
            test_data = loadmat(osp.join(self._root_dir, 'frame_test.mat'))
            test_data = test_data['image_index_test'].squeeze()
            test = []
            for index, item in enumerate(test_data):
                test.append(str(item[0]))
            return test

    def _load_probes(self):
        """
        Load the list of (img, roi) for probes. For test split, it's defined
        by the protocol. For training split, will randomly choose some samples
        from the gallery as probes.
        """
        self.probe_num = 2057
        probes = []
        roi = np.zeros([self.probe_num, 4],  dtype = np.int32)
        probetxt = open(osp.join(self._root_dir, 'query_info.txt'), 'r')
        try:
            i = 0
            for line in probetxt:
                line = line.strip('\r\n')
                pid, x, y, w, h, img_name = line.split(' ')
                roi[i][0] = float(x)
                roi[i][1] = float(y)
                roi[i][2] = float(w)
                roi[i][3] = float(h)
                roi[:, 2:] += roi[:, :2]
                i += 1
                im_name = osp.join(self._data_path, str(img_name)+'.jpg')
                probes.append((im_name, roi, pid))
        finally:
            probetxt.close()

        return probes 
                
    def evaluate_detections(self, gallery_det, det_thresh=0.5, iou_thresh=0.5, 
                                labeled_only = False):
        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        det_thresh (float): filter out gallery detections whose scores below this
        iou_thresh (float): treat as true positive if IoU is above this threshold
        labeled_only (bool): filter out unlabeled background people
        """
        assert self.num_images == len(gallery_det)

        gt_roidb = self.gt_roidb()
        y_true, y_socre = [], []
        count_gt, count_tp = 0, 0
        for gt, det in zip(gt_roidb, gallery_det):
            gt_boxes = gt['boxes']
            if labeled_only:
                inds = np.where(gt['gt_pids'].ravel() > 0)[0]
                if len(inds) == 0: continue 
                gt_boxes = gt_boxes[inds]
            det = np.asarray(det)
            inds = np.where(det[:, 4].ravel() >= det_thresh)[0]
            det = det[inds]
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
            if num_det == 0:
                count_gt += num_gt 
                continue 
            ious = np.zeros((num_gt, num_det), dtype=np.float32)
            for i in xrange(num_gt):
                for j in xrange(num_det):
                    ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
            tfmat = (ious >= iou_thresh)
            # for each det, keep only the largest iou of all the gt
            for j in xrange(num_det):
                largest_ind = np.argmax(ious[:, j])
                for i in xrange(num_gt):
                    if i != largest_ind:
                        tfmat[i, j] = False 
            # for each gt, keep only the largest iou of all the det
            for i in xrange(num_gt):
                largest_ind = np.argmax(ious[i, :])
                for j in xrange(num_det):
                    if j != largest_ind:
                        tfmat[i, j] = False
            for j in xrange(num_det):
                y_score.append(det[j, -1])
                if tfmat[:, j].any():
                    y_true.append(True)
                else:
                    y_true.append(False)
            count_tp += tfmat.sum()
            count_gt += num_gt

        det_rate = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * det_rate
        precision, recall, __ = precision_recall_curve(y_true, y_score)
        recall *= det_rate

        print '{} detection:'.format('labeled only' if labeled_only else
                                     'all')
        print '  recall = {:.2%}'.format(det_rate)
        if not labeled_only:
            print '  ap = {:.2%}'.format(ap)

    def evaluate_search(self, gallery_det, gallery_feat, probe_feat,
                            det_thresh=0.5, gallery_size=-1, dump_json=None):
        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        gallery_feat (list of ndarray): n_det x D features per image
        probe_feat (list of ndarray): D dimensional features per probe image
        det_thresh (float): filter out gallery detections whose scores below this
        dump_json (str): Path to save the results as a JSON file or None
        """
        
        assert self.num_images == len(gallery_det)
        assert self.num_images == len(gallery_feat)
        assert len(self.probes) == len(probe_feat)
        assert gallery_size == -1 
        
        # mapping from gallery image to (det, feat)
        name_to_det_feat = {}
        for name, det, feat in zip(self._image_index,
                            gallery_det, gallery_feat):
            scores = det[:, 4].ravel()
            inds = np.where(scores >= det_thresh)[0]
            if len(inds) > 0:
                name_to_det_feat[name] = (det[inds], feat[inds])

        aps = []
        accs = []
        topk = [1, 5, 10]
        ret = {'image_root': self._data_path, 'results': []}
        for i in xrange(len(self._probes)):
            y_true, y_score = [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0
            # Get L2-normalized feature vector
            feat_p = probe_feat[i].ravel()
            # Ignore the probe image
            probe_imname = str(self._probes[i][0])
            probe_roi = self._probes[i][1]
            probe_gt = []
            probe_label = int(self._probes[i][2])
            tested = set([probe_imname])
            for j in xrange(len(self._image_index)):
                gallery_imname = str(self._image_index[i])
                if gallery_imname in tested:
                    continue 
                count_gt += self._is_contain_gt(probe_label, gallery_imname)
                det, feat_g = name_to_det_feat[gallery_imname]
                # get L2-normalized feature metrix N*D
                assert feat_g.size == np.prod(feat_g.shape[:2])
                feat_g = feat_g.reshape(feat_g.shape[:2])
                # compute cosine similarities
                sim = feat_g.dot(feat_p).ravel()
                # assign label for each det
                label = np.zeros(len(sim), dtype=np.int32)
                if gt.size > 0:
                    w, h = gt[2], gt[3]
                    gt[2:] += gt[:2]
                    probe_gt.append({'img': str(gallery_imname),
                                     'roi': map(float, list(gt))})
                    iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]
                    # only set the first matched det as true positive
                    for j, roi in enumerate(det[:, :4]):
                        if _compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break
                y_true.extend(list(label))
                y_score.extend(list(sim))
                imgs.extend([gallery_imname] * len(sim))
                rois.extend(list(det))

            

if __name__ == '__main__':
    from datasets.pswdb import pswdb 
    d = pswdb('train')
    res = d.roidb 
    from IPython import embed; embed()


















                
























