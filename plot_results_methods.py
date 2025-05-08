import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import utils.plots as plot_utils
import os


PROBLEMS = [
    # "redox-mer",
    # 'solvation', 'kinase',
    # "laser",
    # 'pce',
    "photoswitch",
]
IS_MAX = {
    "redox-mer": False,
    "solvation": False,
    "kinase": False,
    "laser": True,
    "pce": True,
    "photoswitch": True,
}
FEATURE_NAMES_BASE = [
    # 'fingerprints',
    # 'molformer',
]
REAL_FEATURE_NAMES_LLM = [
    # 'gpt2-medium',
    # 'llama-2-7b',
    # "t5-base",
    "t5-base-chem",
]
REAL_FEATURE_NAMES = FEATURE_NAMES_BASE + REAL_FEATURE_NAMES_LLM
METHODS = [
    # 'random',
    # 'gp',
    # 'laplace',
    # 'vi',
    'ensembles',
    'mcdropout',
]
PROBLEM2TITLE = {
    "redox-mer": "Redoxmer (1407)",
    "solvation": "Solvation",
    "kinase": "Kinase",
    "laser": "Laser (10000)",
    "pce": "Photovoltaics",
    "photoswitch": "Photoswitches (392)",
}
PROBLEM2LABEL = {
    "redox-mer": "Redox Potential",
    "solvation": "Solvation Energy",
    "kinase": "Docking Score",
    "laser": "Strength",
    "pce": "PCE",
    "photoswitch": "Wavelength",
}
METHOD2LABEL = {
    "random": "RS", 
    "gp": "GP", 
    "laplace": "LA",
    "vi": "VI",
    "ensembles": "ENS",
    "mcdropout": "MCD"
}
FEATURE2LABEL = {
    "fingerprints": "FP",
    "molformer": "MolFormer",
    "gpt2-medium": "GPT2-M",
    "gpt2-large": "GPT2-L",
    "llama-2-7b": "LL2-7B",
    "t5-base": "T5",
    "t5-base-chem": "T5-Chem",
}
FEATURE2COLOR = {
    "random-fingerprints": (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    "gp-fingerprints": (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    "laplace-fingerprints": (
        0.5490196078431373,
        0.33725490196078434,
        0.29411764705882354,
    ),
    "laplace-molformer": (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    "laplace-t5-base": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    "laplace-gpt2-medium": (
        0.17254901960784313,
        0.6274509803921569,
        0.17254901960784313,
    ),
    "laplace-llama-2-7b": (1.0, 0.4980392156862745, 0.054901960784313725),
    "laplace-t5-base-chem": (
        0.12156862745098039,
        0.4666666666666667,
        0.7058823529411765,
    ),
    "vi-t5-base-chem": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    "ensembles-t5-base-chem": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    "mcdropout-t5-base-chem": (1.0, 0.4980392156862745, 0.054901960784313725),
}

METHOD2COLOR = {
    "laplace": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    "vi": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    "ensembles": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    "mcdropout": (1.0, 0.4980392156862745, 0.054901960784313725)
}

RANDSEEDS = [42] # [1, 2, 3, 4, 5]

# Map methods to their corresponding filenames
METHOD2FILENAME = {
    "ensembles": "just-smiles_trace_best_y_10_ts_ensembles",
    "mcdropout": "just-smiles_trace_best_y_10_ts_mcdropout"
}

YLABEL = "Wavelength"

FIG_WIDTH = 1
FIG_HEIGHT = 0.3
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(
    FIG_WIDTH, FIG_HEIGHT, single_col=False
)
plt.rcParams.update(rc_params)

fig, axs = plt.subplots(1, 3, sharex=True, sharey=False, constrained_layout=True)  # Changed to 1 row
fig.set_size_inches(fig_width, fig_height * 0.5)  # Adjusted height since only one row

for col_idx, (problem, ax) in enumerate(zip(PROBLEMS, axs)):
    if problem is None:
        continue

    # Plot each method
    for method in METHODS:
        print(f"Processing {problem} with {method}")
        real_feature_name = "t5-base-chem"
        path = f"results/{problem}/finetuning/{real_feature_name}"

        # Get the correct filename for this method
        fname = METHOD2FILENAME[method]
        
        MAX_T = 50
        trace_best_y = np.zeros(shape=[len(RANDSEEDS), MAX_T])
        for i, rs in enumerate(RANDSEEDS):
            file_path = f"{path}/{fname}_{rs}.npy"
            print(f"Loading file: {file_path}")
            try:
                arr = np.load(file_path)[:MAX_T]
                trace_best_y[i, : len(arr)] = arr
                print(f"Data loaded for {method}: {arr}")
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue

        mean = np.mean(trace_best_y, axis=0)[1:]  # Over randseeds
        sem = st.sem(trace_best_y, axis=0)[1:]  # Over randseeds

        idx_last = MAX_T - 1
        T = np.arange(1, idx_last + 1)
        c = METHOD2COLOR[method]  # Use method color directly
        print(f"Plotting for {method}: shape T={T.shape}, mean={mean[:idx_last].shape}")
        
        ax.plot(
            T,
            mean[:idx_last],
            c=c,
            ls="solid",
            label=f"{METHOD2LABEL[method]}",
        )
        ax.fill_between(
            T, (mean - sem)[:idx_last], (mean + sem)[:idx_last], color=c, alpha=0.25
        )

    ax.set_title(f"{PROBLEM2TITLE[problem]}")
    ax.set_xlabel(r"$t$")
    
    if col_idx == 0:
        ax.set_ylabel(YLABEL)

    ax.set_xlim(1, 50)

# Create legend
handles, labels = [], []
for method in METHODS:
    line = plt.Line2D([0], [0], color=METHOD2COLOR[method], lw=2)
    handles.append(line)
    labels.append(METHOD2LABEL[method])

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0, 1, 1, 0.075),
    ncols=4,
    handlelength=2,
)

# Save results
path = "../paper/figs"
if not os.path.exists(path):
    os.makedirs(path)

fname = "finetuning_timing"
plt.savefig(f"{path}/{fname}.pdf", bbox_inches="tight")