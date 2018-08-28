import caffe
import numpy as np
DEBUG = False

class RemoveJunkLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom)!=2:
            raise Exception("Just need feat and id_labels")
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)
        top[1].reshape(*bottom[1].data.shape)
    def forward(self, bottom, top):
        feat = bottom[0].data
        labels = bottom[1].data
        ind = np.where(labels!=-1)[0]
        top_feat = feat[ind]
        top[0].reshape(*top_feat.shape)
        top[0].data[...] = top_feat
        top_label = labels[ind]
        top[1].reshape(*top_label.shape)
        top[1].data[...] = top_label
    def backward(self, top, propagate_down, bottom):
        for j in range(2):
            if not propagate_down[j]:
                continue
            else:
                labels = bottom[1].data
                ind = np.where(labels!=-1)[0]
                ind_junk = np.where(labels == -1)[0]
                assert len(ind)==top[0].diff.shape[0]
                for i in range(len(ind)):
                    bottom[j].diff[ind[i]] = top[0].diff[i]
                bottom[j].diff[ind_junk] = 0