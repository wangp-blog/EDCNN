## [Efficient In-loop Filtering Based on Enhanced Deep Convolutional Neural Networks for HEVC](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9049421)

​	We propose an efficient in-loop filtering algorithm based on the enhanced deep convolutional neural networks (EDCNN) for significantly improving the performance of in-loop filtering in HEVC. Based on the statistical analyses, the EDCNN is proposed for efficiently eliminating the artifacts, which adopts three solutions, including **a weighted normalization method**, **a feature information fusion block**, and **a precise loss function**. 

### The in-loop filtering in HEVC

![image-20200520101835115](network/20200520103152.png)

### Our proposed EDCNN

#### The structure of proposed feature information fusion block 

![20200520103208.png](network/20200520103208.png)

![image-20200520101958149](network/20200520103205.png)



#### Test instruction using pretrained model

```
python3 predict.py --model [pretrained model] --dir_demo [demo images directory] --save_name [directory to save] --pre_train [weightfile]
```

#### Arguments
- n_threads: number of threads for data loading
- cpu: use cpu only
- dir_demo: demo image directory
- model: model name
- pre_train: pretrained model directory
- save_name: directory to save