import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda,MaxPooling2D,concatenate, UpSampling2D,Add,Conv1D,GlobalAveragePooling2D, Reshape, Dense, multiply,  Permute
from keras.layers.advanced_activations import LeakyReLU
# from keras.layers import GlobalMaxPooling2D
from dwt_layer import DWT,IWT


"""
    Some code are copied from Souza, Roberto, R. Marc Lebel, and Richard Frayne.
    "A Hybrid, Dual Domain, Cascade of Convolutional Neural Networks for Magnetic Resonance Image Reconstruction." 
    Proceedings of Machine Learning Research–XXXX 1 (2019): 11.
    https://github.com/rmsouza01/CD-Deep-Cascade-MR-Reconstruction
    The data is collected on https://sites.google.com/view/calgary-campinas-dataset
"""

def ifft_layer(kspace):
    real = Lambda(lambda kspace : kspace[:,:,:,0])(kspace)
    imag = Lambda(lambda kspace : kspace[:,:,:,1])(kspace)
    kspace_complex = K.tf.complex(real,imag)
    rec1 = K.tf.abs(K.tf.ifft2d(kspace_complex))
    rec1 = K.tf.expand_dims(rec1, -1)
    return rec1
    
def fft_layer(image):
    """
    Input: 2-channel array representing image domain complex data
    Output: 2-channel array representing k-space complex data
    """
    # get real and imaginary portions
    real = Lambda(lambda image: image[:, :, :, 0])(image)
    imag = Lambda(lambda image: image[:, :, :, 1])(image)

    image_complex = K.tf.complex(real, imag)  # Make complex-valued tensor
    kspace_complex = K.tf.fft2d(image_complex)

    # expand channels to tensorflow/keras format
    real = K.tf.expand_dims(K.tf.real(kspace_complex), -1)
    imag = K.tf.expand_dims(K.tf.imag(kspace_complex), -1)

    # generate 2-channel representation of k-space
    kspace = K.tf.concat([real, imag], -1)
    return kspace
def ifft_layer_2c(kspace_2channel):
    """
    Input: 2-channel array representing k-space
    Output: 2-channel array representing image domain
    """
    # get real and imaginary portions
    real = Lambda(lambda kspace_2channel: kspace_2channel[:, :, :, 0])(kspace_2channel)
    imag = Lambda(lambda kspace_2channel: kspace_2channel[:, :, :, 1])(kspace_2channel)

    kspace_complex = K.tf.complex(real, imag)  # Make complex-valued tensor
    image_complex = K.tf.ifft2d(kspace_complex)

    # expand channels to tensorflow/keras format
    real = K.tf.expand_dims(K.tf.real(image_complex), -1)
    imag = K.tf.expand_dims(K.tf.imag(image_complex), -1)

    # generate 2-channel representation of image domain
    image_complex_2channel = K.tf.concat([real, imag], -1)
    return image_complex_2channel
    
def nrmse(y_true, y_pred):
    denom = K.sqrt(K.mean(K.square(y_true), axis=(1,2,3)))
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=(1,2,3)))\
    /denom

def DC_block_2c_i(rec,mask,sampled_kspace):
    """
    :param rec: Reconstructed data, can be k-space or image domain
    :param mask: undersampling mask
    :param sampled_kspace:
    :param kspace: Boolean, if true, the input is k-space, if false it is image domain
    :return: k-space after data consistency
    """

    rec_kspace = Lambda(fft_layer)(rec)
    rec_kspace_dc = Lambda(lambda rec_kspace : rec_kspace*mask)(rec_kspace)
    rec_kspace_dc = Add()([rec_kspace_dc,sampled_kspace])
    rec = Lambda(ifft_layer_2c)(rec_kspace_dc)
    return rec
def DC_block_2c(rec,mask,sampled_kspace):
    """
    :param rec: Reconstructed data, can be k-space or image domain
    :param mask: undersampling mask
    :param sampled_kspace:
    :param kspace: Boolean, if true, the input is k-space, if false it is image domain
    :return: k-space after data consistency
    """

    # if kspace:
    #     rec_kspace = rec
    # else:
    rec_kspace = Lambda(fft_layer)(rec)
    rec_kspace_dc = Lambda(lambda rec_kspace : rec_kspace*mask)(rec_kspace)
#     rec_kspace_dc =  Multiply()([rec_kspace,mask])
    rec_kspace_dc = Add()([rec_kspace_dc,sampled_kspace])
    return rec_kspace_dc
def cnn_block_02(cnn_input, depth, nf, kshape):
    """
    :param cnn_input: Input layer to CNN block
    :param depth: Depth of CNN. Disregarding the final convolution block that goes back to
    2 channels
    :param nf: Number of filters of convolutional layers, except for the last
    :param kshape: Shape of the convolutional kernel
    :return: 2-channel, complex reconstruction
    """
    layers = [cnn_input]

    for ii in range(depth):
        # Add convolutional block
        layers.append(Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2),padding='same')(layers[-1]))#LeakyReLU(alpha=0.1)
    final_conv = Conv2D(2, (1, 1), activation='linear')(layers[-1])
    rec1 = Add()([final_conv,cnn_input])
    return rec1
def squeeze_excite_block(input, ratio=4):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def uca_block(cnn_input, nf, kshape):
    """
    sub-model of CCA
    :param cnn_input: Input layer to CNN block
    :return: 2-channel, complex reconstruction
    """

    init = cnn_input
    # shortcut = DWT()(init)
    # channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    # filters = init._keras_shape[channel_axis]
    x1 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(init)
    x1 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(x1)
    x1 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(x1)
    pool1=MaxPooling2D()(x1)
    x2 = Conv2D(nf*2, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(pool1)
    x2 = Conv2D(nf*2, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(x2)
    x2 = Conv2D(nf*2, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(x2)
    pool2 = MaxPooling2D()(x2)
    x3 = Conv2D(nf*4, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(pool2)
    x3 = Conv2D(nf*4, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(x3)
    x3 = Conv2D(nf*4, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(x3)
    x3 =squeeze_excite_block(x3)
    up1 =UpSampling2D()(x3)
    up1 = Conv2D(nf * 2, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(up1)
    x4 =concatenate([up1, x2], axis=-1)
    x4 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(x4)
    x4 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(x4)
    x4 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(x4)
    x4 = squeeze_excite_block(x4)
    up2 = UpSampling2D()(x4)
    up2 = Conv2D(nf , kshape, activation=LeakyReLU(alpha=0.2), padding='same')(up2)
    x5 = concatenate([up2, x1], axis=-1)
    x5 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(x5)
    x5 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(x5)
    x5 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(x5)
    x5 = squeeze_excite_block(x5)
    final_conv = Conv2D(2, (1, 1), activation='linear')(x5)
    rec1 = Add()([final_conv, cnn_input])
    return rec1
    
    
def wnet_cca_cscasde_5_32(mu1, sigma1, mu2, sigma2, mask, H=256, W=256, channels=2, kshape=(3, 3), kshape2=(3, 3)):
    #  model of CCA
    inputs = Input(shape=(H, W, channels))
    inputs1 = Lambda(lambda inputs: (inputs * sigma1 + mu1))(inputs)
    ifft1 = Lambda(ifft_layer_2c)(inputs1)
    ifft1 = Lambda(lambda ifft1: (ifft1 - mu2) / sigma2)(ifft1)
    #     kspace_flag = False
    # cnk1 = cnn_block_02(inputs1, 3, 32, kshape2) #k空间网络第一级

    cn1 = uca_block(ifft1, 32, kshape2)
    print('cn1', cn1.shape)
    dc1 = DC_block_2c(cn1, mask, inputs1)
    ifft2 = Lambda(ifft_layer_2c)(dc1)
    print('ifft2', ifft2.shape)
    cn2 = uca_block(ifft2,  32, kshape2)
    dc2 = DC_block_2c(cn2, mask, inputs1)
    ifft3 = Lambda(ifft_layer_2c)(dc2)
    cn3 = uca_block(ifft3, 32, kshape2)
    dc3 = DC_block_2c(cn3, mask, inputs1)
    ifft4 = Lambda(ifft_layer_2c)(dc3)
    cn4 = uca_block(ifft4,  32, kshape2)
    dc4 = DC_block_2c(cn4, mask, inputs1)
    ifft5 = Lambda(ifft_layer_2c)(dc4)
    cn5 = uca_block(ifft5,  32, kshape2)
    cn5 =Add()([cn5, ifft1])
    dc6 = DC_block_2c(cn5, mask, inputs1)
    ifft6 = Lambda(ifft_layer_2c)(dc6)
    res1_scaled=dc6
    out2 = ifft6
    print('o2', out2.shape)
    model = Model(inputs=inputs, outputs=[res1_scaled, out2])
    return model
def wnet_dccnn_5_64(mu1, sigma1, mu2, sigma2, mask, H=256, W=256, channels=2, kshape=(3, 3), kshape2=(3, 3)):
    #  #  model of DC-CNN
    inputs = Input(shape=(H, W, channels))
    inputs1 = Lambda(lambda inputs: (inputs * sigma1 + mu1))(inputs)
    ifft1 = Lambda(ifft_layer_2c)(inputs1)
    ifft1 = Lambda(lambda ifft1: (ifft1 - mu2) / sigma2)(ifft1)

    cn1 = cnn_block_02(ifft1, 5, 64, kshape2)
    print('cn1', cn1.shape)
    dc1 = DC_block_2c(cn1, mask, inputs1)

    ifft2 = Lambda(ifft_layer_2c)(dc1)

    print('ifft2', ifft2.shape)
    cn2 = cnn_block_02(ifft2, 5, 64, kshape2)
    dc2 = DC_block_2c(cn2, mask, inputs1)

    ifft3 = Lambda(ifft_layer_2c)(dc2)
    cn3 = cnn_block_02(ifft3, 5, 64, kshape2)
    dc3 = DC_block_2c(cn3, mask, inputs1)
    ifft4 = Lambda(ifft_layer_2c)(dc3)
    cn4 = cnn_block_02(ifft4, 5, 64, kshape2)
    dc4 = DC_block_2c(cn4, mask, inputs1)
    ifft5 = Lambda(ifft_layer_2c)(dc4)
    cn5 = cnn_block_02(ifft5, 5, 64, kshape2)
    dc6 = DC_block_2c(cn5, mask, inputs1)
    ifft6 = Lambda(ifft_layer_2c)(dc6)

    res1_scaled=dc6
    out2 = ifft6
    print('o2', out2.shape)
    model = Model(inputs=inputs, outputs=[res1_scaled, out2])
    return model
def CA_1d_layer(input, k_size):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, filters )
    se_shape2 = (1, 1,filters)
    se = GlobalAveragePooling2D()(init)
    print("se shape", se.shape)
    se = Reshape(se_shape)(se)


    se = Conv1D(filters, k_size,padding='same', activation='sigmoid')(se)
    se=Reshape(se_shape2)(se)
    print("se 2 shape", se.shape)


    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    
    return  multiply([input, se])
def cnn_block_02_2in_dense_block_5stage_ca1d_dwt_noshort01_catfirst(cnn_input, nf, kshape):
    """
    :param cnn_input: Input layer to CNN block
    :param depth: Depth of CNN. Disregarding the final convolution block that goes back to
    2 channels
    :param nf: Number of filters of convolutional layers, except for the last
    :param kshape: Shape of the convolutional kernel
    :return: 2-channel, complex reconstruction
    """

    init=cnn_input
    x0 = Conv2D(8, (1,1), activation=LeakyReLU(alpha=0.2), padding='same')(init)
    shortcut = DWT()(x0)
    x1 =Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(shortcut)

    x2 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2),dilation_rate=2, padding='same')(x1)
    merge2 = concatenate([ x1, x2], axis=-1)
    merge2 = Conv2D(nf, (1, 1), activation=LeakyReLU(alpha=0.2), padding='same')(merge2)

    x3 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), dilation_rate=3,padding='same')(merge2)
    merge3 = concatenate([ x1, x2, x3], axis=-1)
    merge3 = Conv2D(nf, (1, 1), activation=LeakyReLU(alpha=0.2), padding='same')(merge3)

    x4 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(merge3)
    merge4 = concatenate([ x1, x2, x3, x4], axis=-1)
    merge4 = Conv2D(nf, (1, 1), activation=LeakyReLU(alpha=0.2), padding='same')(merge4)
    x5 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(merge4)
    merge5 = concatenate([ x1, x2, x3, x4, x5], axis=-1)
    merge5 = Conv2D(nf, (1, 1), activation=LeakyReLU(alpha=0.2), padding='same')(merge5)

    ca1d =CA_1d_layer(merge5, 3)
    d1 =Conv2D(8, kshape, activation=LeakyReLU(alpha=0.2), padding='same', kernel_initializer='he_normal')(ca1d)
    iwt1 =IWT()(d1)

    return iwt1
def cnn_block_02_dense_block_5stage_ca1d_dwt_noshortcut01(cnn_input, nf, kshape):
    """
    :param cnn_input: Input layer to CNN block
    :param depth: Depth of CNN. Disregarding the final convolution block that goes back to
    2 channels
    :param nf: Number of filters of convolutional layers, except for the last
    :param kshape: Shape of the convolutional kernel
    :return: 2-channel, complex reconstruction
    """

    init=cnn_input
    shortcut = DWT()(init)
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = shortcut._keras_shape[channel_axis]
    x1 =Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(shortcut)


    x2 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2),dilation_rate=2, padding='same')(x1)
    merge2 = concatenate([ x1, x2], axis=-1)
    merge2 = Conv2D(nf, (1, 1), activation=LeakyReLU(alpha=0.2), padding='same')(merge2)

    x3 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), dilation_rate=3,padding='same')(merge2)
    merge3 = concatenate([ x1, x2, x3], axis=-1)
    merge3 = Conv2D(nf, (1, 1), activation=LeakyReLU(alpha=0.2), padding='same')(merge3)

    x4 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(merge3)
    merge4 = concatenate([ x1, x2, x3, x4], axis=-1)
    merge4 = Conv2D(nf, (1, 1), activation=LeakyReLU(alpha=0.2), padding='same')(merge4)
    x5 = Conv2D(nf, kshape, activation=LeakyReLU(alpha=0.2), padding='same')(merge4)
    merge5 = concatenate([ x1, x2, x3, x4, x5], axis=-1)
    merge5 = Conv2D(nf, (1, 1), activation=LeakyReLU(alpha=0.2), padding='same')(merge5)
        # print('2nd', layers[-1].shape)
    ca1d =CA_1d_layer(merge5, 5)
    d1 =Conv2D(filters, kshape, activation=LeakyReLU(alpha=0.2), padding='same', kernel_initializer='he_normal')(ca1d)
    iwt1 =IWT()(d1)
    final_conv = Conv2D(2, (1, 1), activation='linear')(iwt1)

    return final_conv
def ddd_c10d10_32_dense_block_ca1d_dwt_noshortcut01_cartfirst(mu1, sigma1, mu2, sigma2, mask, H=256, W=256, channels=2, kshape=(3, 3), kshape2=(3, 3)):
    #  model of DCWN

    inputs = Input(shape=(H, W, channels))
    inputs1 = Lambda(lambda inputs: (inputs * sigma1 + mu1))(inputs)
    ifft1 = Lambda(ifft_layer_2c)(inputs1)
    ifft1 = Lambda(lambda ifft1: (ifft1 - mu2) / sigma2)(ifft1)

    # ifft1 = Lambda(lambda ifft2: (ifft2 - mu2) / sigma2)(ifft1)
    cni1 = cnn_block_02_dense_block_5stage_ca1d_dwt_noshortcut01(ifft1,  32, kshape2) #图像空间网络第一级
    print('cn1', cni1.shape)
    dci1 = DC_block_2c_i(cni1, mask, inputs1) #这个输出均为image
    
    cni2 =cnn_block_02_2in_dense_block_5stage_ca1d_dwt_noshort01_catfirst(concatenate([dci1, ifft1], axis=-1),  32, kshape2)
    print('cn2', cni2.shape)
    dci2 = DC_block_2c_i(cni2, mask, inputs1)  # 这个输出均为image

    cni3 = cnn_block_02_2in_dense_block_5stage_ca1d_dwt_noshort01_catfirst(concatenate([dci1, ifft1,dci2], axis=-1),  32, kshape2)
    print('cn3', cni3.shape)
    dci3 = DC_block_2c_i(cni3, mask, inputs1)  # 这个输出均为image

    cni4 = cnn_block_02_2in_dense_block_5stage_ca1d_dwt_noshort01_catfirst(concatenate([dci1, ifft1,dci2,dci3], axis=-1), 32, kshape2)
    print('cn4', cni4.shape)
    dci4 = DC_block_2c_i(cni4, mask, inputs1)  # 这个输出均为image

    cni5 = cnn_block_02_2in_dense_block_5stage_ca1d_dwt_noshort01_catfirst(concatenate([dci1, ifft1,dci2,dci3,dci4], axis=-1),  32, kshape2)
    print('cn5', cni5.shape)
    dci5 = DC_block_2c_i(cni5, mask, inputs1)  # 这个输出均为image

    cni6 =cnn_block_02_2in_dense_block_5stage_ca1d_dwt_noshort01_catfirst(concatenate([dci1, ifft1,dci2,dci3,dci4,dci5], axis=-1),  32, kshape2)
    print('cn6', cni6.shape)
    dci6 = DC_block_2c_i(cni6, mask, inputs1)  # 这个输出均为image

    cni7 = cnn_block_02_2in_dense_block_5stage_ca1d_dwt_noshort01_catfirst(concatenate([dci1, ifft1,dci2,dci3,dci4,dci5,dci6], axis=-1),32, kshape2)
    print('cn7', cni7.shape)
    dci7 = DC_block_2c_i(cni7, mask, inputs1)  # 这个输出均为image
 
    cni8 = cnn_block_02_2in_dense_block_5stage_ca1d_dwt_noshort01_catfirst(concatenate([dci1, ifft1,dci2,dci3,dci4,dci5,dci6,dci7], axis=-1),  32, kshape2)
    print('cn8', cni8.shape)
    dci8 = DC_block_2c_i(cni5, mask, inputs1)  # 这个输出均为image


    cni9 = cnn_block_02_2in_dense_block_5stage_ca1d_dwt_noshort01_catfirst(concatenate([dci1, ifft1,dci2,dci3,dci4,dci5,dci6,dci7,dci8], axis=-1),  32, kshape2)
    print('cn9', cni9.shape)
    dci9 = DC_block_2c_i(cni9, mask, inputs1)  # 这个输出均为image


    cni10 = cnn_block_02_2in_dense_block_5stage_ca1d_dwt_noshort01_catfirst(concatenate([dci1, ifft1,dci2,dci3,dci4,dci5,dci6,dci7,dci8,dci9], axis=-1),  32, kshape2)
    print('cn10', cni10.shape)
    dci10 = DC_block_2c_i(cni10, mask, inputs1)  # 这个输出均为image


    fft6 = Lambda(fft_layer)(dci10)  # 这个输出为k空间


    res1_scaled=fft6
    out2 = dci10
    # out2 = Lambda(lambda out2 : (out2  - mu2) / sigma2)(out2 )
    print('o2', out2.shape)
    model = Model(inputs=inputs, outputs=[res1_scaled, out2])
    return model
