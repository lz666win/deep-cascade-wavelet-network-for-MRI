# deep-cascade-wavelet-network-for-MRI
deep cascade wavelet network for CS-MRI.

Some code are copied from Souza, Roberto, R. Marc Lebel, and Richard Frayne.
"A Hybrid, Dual Domain, Cascade of Convolutional Neural Networks for Magnetic Resonance Image Reconstruction." 

https://github.com/rmsouza01/CD-Deep-Cascade-MR-Reconstruction

The data is collected on https://sites.google.com/view/calgary-campinas-dataset

you should download the data, then located them to the corresponding folder.

How to use:

make a folder: Test, then put e14498s5_P60928.7.npy into this folder.

make a folder: Models, then put 3 .hdf5 files into this folder.

make a folder: Data, put stats_fs_unet_norm_2c_2020.npy into this folder,then you can run test_calgary_campinas_cart_5.ipynb to see results.
