# MCTSteg

Implementation of our TIFS paper (DOI (identifier) 10.1109/TIFS.2021.3104140)

"MCTSteg: A Monte Carlo Tree Search-based Reinforcement Learning Framework for Universal Non-additive Steganography"

This program is stable under Python=2.7!

We recommend that using conda before installing all the requirements. The details of our local conda environment are in:

environment.yaml
If your local dependencies are the same as us, then you can run this command to setup your environment:

conda env create -f environment.yaml
If not, you can first create a python2.7 environment and running this command:

pip install -r requirement.txt
Please setup Matlab interface for Python!!!

This procedure is various in different operating systems, please check the corresponding tutorial on official Matlab website!

If Matlab is correctly connected, you can use following pakages in your python environment:

import matlab
import matlab.engine
Directories and files included in the implementation:

'datas/' - Some of the images of our experiment.

'libs/' - Functional code.

'models' - Environmental Model used in MCTSteg.

'JPEG_Toolbox' - Reading and Writing scripts for JPEG image.

'libs/quant_tables' - MAT files of JPEG quantization matrices.

To run MCTSteg in spacial domain, please use this command:

python main_spacial.py -p ./model/SZU_SRNet_Spacial/Model_420000.ckpt -s 128 -a 1.5
To run MCTSteg in JPEG domain, please use this command:

python main_jpeg.py -p ./model/SZU_SRNet_JPEG/Model_560000.ckpt -s 128 -a 1.5
