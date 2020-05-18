import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from PIL import Image
sys.path.append("/home/ramial-maskari/Documents/Pytorch Network")
from utilities.util import read_meta_dict

def load_image(name, axis,image_input_path):
    item_path = f"{image_input_path}/{name}/{name}_{axis}.tiff"
    img = Image.open(item_path)
    return np.array(img)

def create_subplot(fig_axis, p_list, index,image_input_path):
    patch = p_list.iloc[index]["patch"]
    axis  = p_list.iloc[index]["axis"]
    prop  = p_list.iloc[index]["propability"]
    img = load_image(patch, axis, image_input_path)
    fig_axis.imshow(img)
    fig_axis.set_title(f"{patch}_{axis}.tiff {prop:.5f}")

def create_histogram(df, path,image_input_path):
    """ Create a figure with 4 Subplots """
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

def create_high_confident_tp(df, path,image_input_path):
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
    create_histogram(df, path, image_input_path)
    create_high_confident_tp(df, path, image_input_path)
    create_high_confident_fp(df, path, image_input_path)
    create_uncertain(df, path, image_input_path)
    plt.gray()
    plt.tight_layout()
