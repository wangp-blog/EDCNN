## [Efficient In-loop Filtering Based on Enhanced Deep Convolutional Neural Networks for HEVC](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9049421)

### The in-loop filtering in HEVC

![image-20200520101835115](network/20200520103152.png)

We propose an efficient in-loop filtering algorithm based on the enhanced deep convolutional neural networks (EDCNN) for significantly improving the performance of in-loop filtering in HEVC.  the EDCNN is proposed for efficiently eliminating the artifacts, which adopts three solutions, including **a weighted normalization method**, **a feature information fusion block**, and **a precise loss function**. 

------

### Our proposed EDCNN

#### 1. The structure of proposed feature information fusion block 

<img src="network/20200520103208.png" alt="20200520103208.png" style="zoom: 20%;" />

#### 2. The architecture of proposed EDCNN 

![image-20200520101958149](network/20200520103205.png)

#### 3. The detailed network parameters

<img src="network/20200520131415.png" alt="20200520131415" style="zoom:33%;" />

------

### Experimental Results

#### 1. The PSNR standard deviations of Low-Delay coding structure

<img src="network/20200520131610.png" alt="20200520131610" style="zoom: 70%;" />

#### 2. The PSNR standard deviations of Random-Access coding structure

![20200520131941](network/20200520131941.png)

#### 3. Video subjective quality comparison

![20200520132202](network/20200520132202.png)

### Test instruction using pretrained model

```python
python3 predict.py --model [pretrained model] --dir_demo [demo images directory] --save_name [directory to save] --pre_train [weightfile]
```

#### Arguments
- n_threads: number of threads for data loading
- cpu: use cpu only
- dir_demo: demo image directory
- model: model name
- pre_train: pretrained model directory
- save_name: directory to save

### Citation

Z. Pan, X. Yi, Y. Zhang, B. Jeon and S. Kwong, "Efficient In-Loop Filtering Based on Enhanced Deep Convolutional Neural Networks for HEVC," in *IEEE Transactions on Image Processing*, vol. 29, pp. 5352-5366, 2020