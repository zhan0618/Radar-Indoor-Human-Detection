import h5py
import matplotlib.pyplot as plt
import numpy as np
import re

# # Open the HDF5 file
# test_file = r'G:\Jiarui\raw_h5\E10_H2_4m_a.h5'
# with h5py.File(test_file, "r") as f:
#     # Load the tensor from the file
#     tensor = f["tensor"][:]

# print(tensor)

# # figure = plt.figure()
# # ax = figure.add_subplot(111)
# # # import pdb
# # # pdb.set_trace()
# # data = tensor[10,:,:]
# # #im = ax.imshow(data, cmap="jet",vmin=Max-45,vmax=Max)
# # im = ax.imshow(data, cmap="jet",vmin=-50,vmax=30)

# # ax.set_aspect(0.75*data.shape[1] / data.shape[0])
# # plt.show()


def save_img(tensor,name):
    figure = plt.figure()
    ax = figure.add_subplot(111)
    data = tensor[16,:,:]
    im = ax.imshow(data, cmap="jet",vmin=-45,vmax=20)
    ax.set_aspect(0.75*data.shape[1] / data.shape[0])
    ax.axis('off')
    plt.savefig(name, bbox_inches='tight')

if __name__ == "__main__":
    test_file = r'G:\Jiarui\Sep14\md_h5_files\E13_H2_6m_b.h5'
    name = "ori"+'.png'
    with h5py.File(test_file, "r") as f:
        # Load the tensor from the file
        tensor = f["tensor"][:]
    save_img(tensor,name)

    test_file1 = r'G:\Jiarui\Sep14\rm_bg_md_h5_files\E13_H2_6m_b.h5'
    name1 = 'rm'+'.png'
    with h5py.File(test_file, "r") as f:
        # Load the tensor from the file
        tensor = f["tensor"][:]
    save_img(tensor,name1)




    
    
    

    