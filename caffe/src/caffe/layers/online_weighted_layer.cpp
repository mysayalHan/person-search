#include <vector>

#include "cafe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template <typename Dtype>
    void OnlineWeightedLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        num_classes_ = this->layer_param_.labeled_matching_param().num_classes();
        momentum_ = this->layer_param_.labeled_matching_param().momentum();
        // Set softmax layer param
        LayerParameter softmax_param(this->layer_param_);
        softmax_param.set_type("Softmax");
        softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
        // Blob are "permuted" into [batch_num, height, width, channel]
        N_ = bottom[0]->shape(0);
        H_ = bottom[0]->shape(1);
        W_ = bottom[0]->shape(2);
        C_ = bottom[0]->count(3);
        // Check if we need to set up the weights
        if (this->blobs_.size() > 0) {
            LOG(INFO) << "Skipping parameter initialization";
        }
        else {
            this->blobs_.resize(1);
            // Intialize the weight
            vector<int> feat_shape(1);
            feat_shape[0] = C_
            this->blobs_[0].reset(new Blob<Dtype>(feat_shape));
            // fill the weights
            shared_ptr<Filler<Dtype> > feat_filler(GetFiller<Dtype>(
                this->layer_param_.online_weighted_param().feat_filler()));
            feat_filler->Fill(this->blobs_[0].get());
        }
        // "Parameters" will be updated, but not by standard backprop with gradients.
        this->param_propagate_down_.resize(this->blobs_.size(), true);
    }

    template <typename Dtype>
    void OnlineWeightedLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        vector<int> mask_shape(4);
        vector<int> top_shape(4);
        mask_shape[0] = N_;
        mask_shape[1] = H_;
        mask_shape[2] = W_;
        mask_shape[3] = 1;
        mask_.Reshape(mask_shape);
        top[0]->Reshape(mask_shape)
    }

    template <typename Dtype>
    void OnlineWeightedLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        //const int M = bottom[0]->shape(0); //batchsize
        //const int K = bottom[0]->count(3); //channel
        //const int N = bottom[0]->count(1,3); //W*H
        Dtype* mask_data = mask_->mutable_cpu_data()
        const Dtype* bottom_label = bottom[1]->cpu_data();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        CHECK_EQ(N_*H_*W_*C_, bottom[0]->count())
            << "Input size incompatible with initialization.";
        // calculate forward mask
        for (int i = 0; i < N_; ++i){
            const int label_value = static_cast<int>(bottom_label[i]);
            if (label_value==num_classes_){
                caffe_set(H_*W_, 0., mask_data+i*H_*W_);
            }
            else{
                for (int j = 0; j < H_*W_; ++j){
                    mask_data[i*H_*W_+j] = caffe_cpu_dot(C_, bottom_data+i*H_*W_+j, this->blob_[0]->cpu_data());
                }
            }
        }
        // Setup softmax layer
        // softmax_bottom_vec_.clear();
        // softmax_bottom_vec_.push_back(mask_);
        // softmax_top_vec_.clear();
        // softmax_top_vec_.push_back(&prob_);
        // softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
        // The forward pass computes the softmax prob values.
        // softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
        // const Dtype* prob_data = prob_.cpu_data();
        // vector<int> top_shape(4);
        // prob_data->Reshape(top_shape);
        caffe_copy(N_*H_*W_*1, top[0]->mutable_cpu_data(), mask_data);
    }

    template <typename Dtype>
    void OnlineWeightedLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& top){
        if (!(!propagate_down[0]&&!propagate_down[1])) {
            // bottom data does not engage back propagation
            LOG(INFO) << "Bottom data does not engage back propagation";
        }
        else {
            if (this->param_propagate_down_[0]) {
                const Dtype* bottom_data = bottom[0]->cpu_data();
                const Dtype* bottom_label = bottom[1]->cpu_data();
                Dtype* weight = this->blobs_[0]->mutable_cpu_data();
                // update instance feature
                for (int i = 0; i < N_; ++i) {
                    const int label_value = static_cast<int>(bottom_label[i]);
                    if (label_value == num_classes_) continue;
                    for (int j = 0; j < H_*W_; j++) {
                        // w <- momentum * w + (1-momentum) * x
                        caffe_cpu_axpby(C_, (Dtype)1. - momentum_, bottom_data + i*H_*W_ + j803,
                            momentum_, weight);
                    }
                }
            }
        }
    }
}