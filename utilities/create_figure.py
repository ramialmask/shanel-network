import os
import sys
sys.path.append("/home/ramial-maskari/Documents/Pytorch Network/")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image
from utilities.util import read_meta_dict

def load_image(name, axis,image_input_path):
    item_path = f"{image_input_path}/{name}/{name}_{axis}.tiff"
    img = Image.open(item_path)
    return np.array(img)

def create_subplot(fig_axis, p_list, index,image_input_path):
    if len(p_list) > 0:
        patch = p_list.iloc[index]["patch"]
        axis  = p_list.iloc[index]["axis"]
        prop  = p_list.iloc[index]["propability"]
        img = load_image(patch, axis, image_input_path)
        fig_axis.imshow(img)
        fig_axis.set_title(f"{patch}_{axis}.tiff {prop:.5f}")

def create_histogram(df, path,image_input_path):
    """ Create a figure with 4 Subplots """
    print("Creating histogram")
    fig, ax = plt.subplots()
    ax.set_title("Histogram of p-values")

    real_heights, real_bins = np.histogram(df.loc[df["predicted class"] == 0.]["propability"])
    syn_heights, syn_bins = np.histogram(df.loc[df["predicted class"] == 1.]["propability"])

    prop_proxy = [df.loc[df["predicted class"] == 0.]["propability"],df.loc[df["predicted class"] == 1.]["propability"]]

    colors=["forestgreen","mediumturquoise"]
    ax.hist(prop_proxy, histtype="bar",color=colors)

    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in colors]
    labels= ["Real Data","Synthetic Data"]
    ax.legend(handles, labels)
    ax.set_xlabel("P-Value")
    ax.set_ylabel("# Images")
    plt.tight_layout()
    plt.savefig(f"{path}histogram.png", dpi=400)

def create_histogram_real(df, path):
    print("Creating histogram for real data")
    fig, ax = plt.subplots()
    ax.set_title("Histogram of p-values for real data")

    prop_real = df.loc[df["class"] == 0.]["propability"]

    colors=["forestgreen"]
    ax.hist(prop_real, histtype="bar",color=colors)

    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in colors]
    labels= ["Real Data"]
    ax.legend(handles, labels)
    ax.set_xlabel("P-Value")
    ax.set_ylabel("# Images")
    plt.tight_layout()
    plt.savefig(f"{path}histogram_real.png", dpi=400)

def create_histogram_syn(df, path):
    print("Creating histogram for synthetic data")
    fig, ax = plt.subplots()
    ax.set_title("Histogram of p-values for synthetic data")

    prop_syn = df.loc[df["class"] == 1.]["propability"]

    colors=["mediumturquoise"]
    ax.hist(prop_syn, histtype="bar",color=colors)

    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in colors]
    labels= ["Synthetic Data"]
    ax.legend(handles, labels)
    ax.set_xlabel("P-Value")
    ax.set_ylabel("# Images")
    plt.tight_layout()
    plt.savefig(f"{path}histogram_syn.png", dpi=400)

def create_high_confident_tp(df, path,image_input_path):
    print("Creating high confident true classified summary")
    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Highly confident correct classified")
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

    high_confident_real = df.loc[(df["predicted class"] == 0.) & (df["class"] == 0.)]
    high_confident_real = high_confident_real.sort_values("propability")
    
    f_ax1 = fig.add_subplot(spec[0, 0])
    create_subplot(f_ax1, high_confident_real, 1, image_input_path)

    f_ax2 = fig.add_subplot(spec[0, 1])
    create_subplot(f_ax2, high_confident_real, 2, image_input_path)

    f_ax3 = fig.add_subplot(spec[0, 2])
    create_subplot(f_ax3, high_confident_real, 3, image_input_path)

    high_confident_syn = df.loc[(df["predicted class"] == 1.) & (df["class"] == 1.)]
    high_confident_syn = high_confident_syn.sort_values("propability", ascending=False)

    f_ax4 = fig.add_subplot(spec[1, 0])
    create_subplot(f_ax4, high_confident_syn, 1, image_input_path)
    
    f_ax5 = fig.add_subplot(spec[1, 1])
    create_subplot(f_ax5, high_confident_syn, 2, image_input_path)
    
    f_ax6 = fig.add_subplot(spec[1, 2])
    create_subplot(f_ax6, high_confident_syn, 3, image_input_path)
    plt.tight_layout()
    plt.savefig(f"{path}certain tc.png",dpi=400)

def create_high_confident_fp(df, path,image_input_path):
    print("Creating high confident false classified summary")
    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Highly confident incorrect classified")
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

    high_confident_real = df.loc[(df["predicted class"] == 0.) & (df["class"] == 1.)]
    high_confident_real = high_confident_real.sort_values("propability")
    
    f_ax1 = fig.add_subplot(spec[0, 0])
    create_subplot(f_ax1, high_confident_real, 1, image_input_path)

    f_ax2 = fig.add_subplot(spec[0, 1])
    create_subplot(f_ax2, high_confident_real, 2, image_input_path)

    f_ax3 = fig.add_subplot(spec[0, 2])
    create_subplot(f_ax3, high_confident_real, 3, image_input_path)

    high_confident_syn = df.loc[(df["predicted class"] == 1.) & (df["class"] == 0.)]
    high_confident_syn = high_confident_syn.sort_values("propability", ascending=False)

    f_ax4 = fig.add_subplot(spec[1, 0])
    create_subplot(f_ax4, high_confident_syn, 1, image_input_path)
    
    f_ax5 = fig.add_subplot(spec[1, 1])
    create_subplot(f_ax5, high_confident_syn, 2, image_input_path)
    
    f_ax6 = fig.add_subplot(spec[1, 2])
    create_subplot(f_ax6, high_confident_syn, 3, image_input_path)
    plt.tight_layout()
    plt.savefig(f"{path}certain fc.png",dpi=400)

def create_uncertain(df, path,image_input_path):
    print("Creating uncertain patches summary")
    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Uncertain patches")
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

    uncertain_real = df.loc[(df["class"] == 0.) & (df["propability"] >= 0.4) & (df["propability"] <= 0.6)]
    
    f_ax1 = fig.add_subplot(spec[0, 0])
    create_subplot(f_ax1, uncertain_real, -1, image_input_path)

    f_ax2 = fig.add_subplot(spec[0, 1])
    create_subplot(f_ax2, uncertain_real, -2, image_input_path)

    f_ax3 = fig.add_subplot(spec[0, 2])
    create_subplot(f_ax3, uncertain_real, -3, image_input_path)

    uncertain_syn = df.loc[(df["class"] == 1.) & (df["propability"] >= 0.4) & (df["propability"] <= 0.6)]

    f_ax4 = fig.add_subplot(spec[1, 0])
    create_subplot(f_ax4, uncertain_syn, -1, image_input_path)
    
    f_ax5 = fig.add_subplot(spec[1, 1])
    create_subplot(f_ax5, uncertain_syn, -2, image_input_path)
    
    f_ax6 = fig.add_subplot(spec[1, 2])
    create_subplot(f_ax6, uncertain_syn, -3, image_input_path)

    plt.tight_layout()
    plt.savefig(f"{path}uncertain.png",dpi=400)

def create_summary(df, path, image_input_path):
    """Creates a summary for a classification crossvalidation training
    - Histogram of the p-values
    - An overview of correctly classified patches with extreme p-values
    - An overview of incorrectly classified patches with extreme p-values
    - An overview of patches with p-values suggesting network uncertainty
    """
    print("Creating visual summary")
    print(df)
    create_histogram(df, path, image_input_path)
    create_histogram_real(df, path)
    create_histogram_syn(df, path)
    create_high_confident_tp(df, path, image_input_path)
    create_high_confident_fp(df, path, image_input_path)
    create_uncertain(df, path, image_input_path)
    plt.gray()
    plt.tight_layout()

# path = "/media/ramial-maskari/16TBDrive/Synthetic Neuron Creation/classification/output/models/Classification 2020 05 30 Data leanclassification2d Adam factor 0.1 WBCELoss LR=1e-5 Blocksize 100 Epochs 100  | 2020-06-01 00:13:21.426606/"
# df = pd.read_csv(path + "test.csv")
# image_input_path = "/media/ramial-maskari/16TBDrive/Synthetic Neuron Creation/classification/input/syn_data_20_05_30_subset"
# create_summary(df, path, image_input_path)
