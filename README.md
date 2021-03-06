# Neural Style Transfer by Gatys et al. Chainer 4.0 
[Chainer v4.0](https://github.com/chainer/chainer) implementation of "Image Style Transfer Using Convolutional Neural Networks(2016)" by Gatys et al.

[Image Style Transfer Using Convolutional Neural Networks. Gatys, L.A. et al(2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) 


 The purpose is further research of Gatsy's article. 
 So source code keeps very simple for modification and for that reason, the sophisticated functions of chainer(i.e. updater, trainer) are not used.
 
 ## Results
 ### Original Image
 hourly bell at Kawagoe city Japan. 
 
<img src="https://farm1.staticflickr.com/886/27469269047_17ef5222d0_b.jpg" width="300" alt="Hour Bell at Kawagoe , Japan"> 

 ### Style 1
 "Starry Night" by Van Gogh.
 
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg" width="500" alt="Starry Night by Van Gogh"> 

 ### Result 1
<img src="https://farm1.staticflickr.com/967/42291281042_3b5b2d0c1c_z.jpg" width="300" alt="Hour Bell at Kawagoe , Japan">
 
 ### Style 2
 "Gardanne" by Paul Cezanne. 
 
 <img src="https://upload.wikimedia.org/wikipedia/commons/a/a8/Paul_Cezanne_Gardanne.jpg" width="500" alt="Gardanne by Paul Cezanne">
 
 ### Result 2
 <img src="https://farm1.staticflickr.com/978/42339298931_6ab769df7d_z.jpg" width="300" alt="Hour Bell at Kawagoe , Japan">

 ## 
 GPU: GeForce GTX1080i 
 
 Elapsed Time: about 12min for generating 600px squared image. 
 
 

# Usage 
## Environment
- python3.5 (3.0+)
- chainer4.0
- cupy4.0.0
- cuda8.0+
- cuDNN6.0+

## Generate Transferred Image 
`python generate.py -i images/tokinokane600.jpg -s [dir_path]/style.png -o [dir_path for output] -g 0` 
 
 **Shape of images must be square.**

## Parameters
See [source](https://github.com/TetsuyaOdaka/style-transfer-gatys/blob/master/generate.py)


# Acknowlegement
thanks to the authors to which I referred.
- chainer-gogh(https://github.com/pfnet-research/chainer-gogh) 
