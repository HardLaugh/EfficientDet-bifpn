this repo is based on mmdetection.  
I don't have much time to do it，emm just for fun

train efficientdet with the same number of iterations as tensorflow-efficientdet from paperv2 in arxiv[https://arxiv.org/abs/1911.09070v2] , results are 2~4 mAP lower   
efficientdet-d0  32.4mAP batchsize=128 epoch=300 (new)  
efficientdet-d1  38.5mAP batchsize=96 epoch=300 (new)  
efficientdet-d2  40.4mAP batchsize=64 epoch=154  
efficientdet-d3  43.6mAP batchsize=32 epoch=115  
efficientdet-d5  47.4mAP batchsize=16 epoch=38  


i did not release code about EMA. it is suggested that you can install mmcv, which already has ema hook

Please let me know any possible improvement
