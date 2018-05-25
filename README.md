# style-transfer-gatys
Chainer v4.0 implementation of "Image Style Transfer Using Convolutional Neural Networks(2016)" by Gatys et al.

[Image Style Transfer Using Convolutional Neural Networks. Gatys, L.A. et al(2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) 
 
 ## results
<img src="https://farm1.staticflickr.com/886/27469269047_17ef5222d0_b.jpg" width="300" alt="Hour Bell at Kawagoe , Japan"> 
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg" width="500" alt="Starry Night by Van Gogh"> 
<img src="https://farm1.staticflickr.com/967/42291281042_3b5b2d0c1c_z.jpg" width="300" alt="Hour Bell at Kawagoe , Japan">
 
 <img src="https://upload.wikimedia.org/wikipedia/commons/a/a8/Paul_Cezanne_Gardanne.jpg" width="500" alt="Gardanne by Paul Cezanne">
 <img src="https://farm1.staticflickr.com/978/42339298931_6ab769df7d_z.jpg" width="300" alt="Hour Bell at Kawagoe , Japan">

# Usage 
## Environment
- python3.5+
- chainer4.0
- cupy4.0.0
- cuda8.0
- cuDNN6.0

## Usage
`python generate.py -i images/tokinokane600.jpg -s styles-all/style-r.png -o /home/odaka/out-test -g 0` 
 
 **Shape of images must be square.**

## Parameters
See [source](https://github.com/TetsuyaOdaka/style-transfer-gatys/blob/master/generate.py)

# Acknowlegement
very appreciate the authors to which I referred.
- chainer-gogh(https://github.com/pfnet-research/chainer-gogh) 
