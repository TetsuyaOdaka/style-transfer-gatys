# style-transfer-gatys
Chainer v4.0 implementation of "Image Style Transfer Using Convolutional Neural Networks(2016)" by Gatys et al.

[Image Style Transfer Using Convolutional Neural Networks. Gatys, L.A. et al(2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) 
 
# Usage 
## Environment
- python3.5+
- chainer4.0
- cupy4.0.0
- cuda8.0
- cuDNN6.0

## Usage
`python generate.py -i images/tokinokane600.jpg -s styles-all/style-r.png -o /home/odaka/out-test -g 0`

## Parameters
See source

# Acknowlegement
very appreciate the authors to which I referred.
- chainer-gogh(https://github.com/pfnet-research/chainer-gogh) 
