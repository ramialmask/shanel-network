import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from PIL import Image

# df = pd.DataFrame(columns=['patch','axis','class','predicted class','false classified','propability','image'])
df = pd.read_csv("/media/ramial-maskari/16TBDrive/Synthetic Neuron Creation/classification/output/df 7 Epochs.csv")


def load_image(name, axis):
    item_path = f"/media/ramial-maskari/16TBDrive/Synthetic Neuron Creation/classification/input/syn_data_20_05_05_subset/{name}/{name}_{axis}.tiff"
    img = Image.open(item_path)
    return np.array(img)



def create_subplot(fig_axis, p_list, index):
    patch = p_list.iloc[index]["patch"]
    axis  = p_list.iloc[index]["axis"]
    prop  = p_list.iloc[index]["propability"]
    img = load_image(patch, axis)
    fig_axis.imshow(img)
    fig_axis.set_title(f"{patch}_{axis}.tiff {prop:.5f}")

def create_histogram(df):
    # Create a figure with 4 Subplots
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
    plt.savefig("/media/ramial-maskari/16TBDrive/Synthetic Neuron Creation/classification/output/histogram 7 Epochs.png", dpi=400)

def create_high_confident_tp(df):
    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Highly confident correct classified")
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

    high_confident_real = df.loc[(df["predicted class"] == 0.) & (df["class"] == 0.)]
    high_confident_real = high_confident_real.sort_values("propability")
    
    f_ax1 = fig.add_subplot(spec[0, 0])
    create_subplot(f_ax1, high_confident_real, 1)

    f_ax2 = fig.add_subplot(spec[0, 1])
    create_subplot(f_ax2, high_confident_real, 2)

    f_ax3 = fig.add_subplot(spec[0, 2])
    create_subplot(f_ax3, high_confident_real, 3)

    high_confident_syn = df.loc[(df["predicted class"] == 1.) & (df["class"] == 1.)]
    high_confident_syn = high_confident_syn.sort_values("propability", ascending=False)

    f_ax4 = fig.add_subplot(spec[1, 0])
    create_subplot(f_ax4, high_confident_syn, 1)
    
    f_ax5 = fig.add_subplot(spec[1, 1])
    create_subplot(f_ax5, high_confident_syn, 2)
    
    f_ax6 = fig.add_subplot(spec[1, 2])
    create_subplot(f_ax6, high_confident_syn, 3)
    plt.tight_layout()
    plt.savefig("/media/ramial-maskari/16TBDrive/Synthetic Neuron Creation/classification/output/certain tc 7 Epochs.png",dpi=400)

def create_high_confident_fp(df):
    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Highly confident incorrect classified")
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

    high_confident_real = df.loc[(df["predicted class"] == 0.) & (df["class"] == 1.)]
    high_confident_real = high_confident_real.sort_values("propability")
    
    f_ax1 = fig.add_subplot(spec[0, 0])
    create_subplot(f_ax1, high_confident_real, 1)

    f_ax2 = fig.add_subplot(spec[0, 1])
    create_subplot(f_ax2, high_confident_real, 2)

    f_ax3 = fig.add_subplot(spec[0, 2])
    create_subplot(f_ax3, high_confident_real, 3)

    high_confident_syn = df.loc[(df["predicted class"] == 1.) & (df["class"] == 0.)]
    high_confident_syn = high_confident_syn.sort_values("propability", ascending=False)

    f_ax4 = fig.add_subplot(spec[1, 0])
    create_subplot(f_ax4, high_confident_syn, 1)
    
    f_ax5 = fig.add_subplot(spec[1, 1])
    create_subplot(f_ax5, high_confident_syn, 2)
    
    f_ax6 = fig.add_subplot(spec[1, 2])
    create_subplot(f_ax6, high_confident_syn, 3)
    plt.tight_layout()
    plt.savefig("/media/ramial-maskari/16TBDrive/Synthetic Neuron Creation/classification/output/certain fc 7 Epochs.png",dpi=400)

def create_uncertain(df):
    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Uncertain patches")
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

    uncertain_real = df.loc[(df["class"] == 0.) & (df["propability"] >= 0.4) & (df["propability"] <= 0.6)]
    
    f_ax1 = fig.add_subplot(spec[0, 0])
    create_subplot(f_ax1, uncertain_real, -1)

    f_ax2 = fig.add_subplot(spec[0, 1])
    create_subplot(f_ax2, uncertain_real, -2)

    f_ax3 = fig.add_subplot(spec[0, 2])
    create_subplot(f_ax3, uncertain_real, -3)

    uncertain_syn = df.loc[(df["class"] == 1.) & (df["propability"] >= 0.4) & (df["propability"] <= 0.6)]

    f_ax4 = fig.add_subplot(spec[1, 0])
    create_subplot(f_ax4, uncertain_syn, -1)
    
    f_ax5 = fig.add_subplot(spec[1, 1])
    create_subplot(f_ax5, uncertain_syn, -2)
    
    f_ax6 = fig.add_subplot(spec[1, 2])
    create_subplot(f_ax6, uncertain_syn, -3)

    plt.tight_layout()
    plt.savefig("/media/ramial-maskari/16TBDrive/Synthetic Neuron Creation/classification/output/uncertain 7 Epochs.png",dpi=400)



create_histogram(df)
create_high_confident_tp(df)
create_high_confident_fp(df)
create_uncertain(df)
print(len(df.loc[(df['class'] != df['predicted class'])]))
plt.gray()
plt.tight_layout()
# plt.show()

