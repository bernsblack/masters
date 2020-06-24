import os

import matplotlib.pyplot as plt
import numpy as np
from pprint import pformat

from matplotlib.axis import Axis
from matplotlib.lines import Line2D

from sparse_discrete_table import SparseDiscreteTable
from utils.mutual_information import construct_mi_grid
from utils.setup import setup
import logging as log

import os

_info = log.info
global MAX_Y_LIM
global normalize


def set_line(f, t, ax, line):
    global MAX_Y_LIM
    t_min = np.min(t)
    t_max = np.max(t)
    t_pad = (t_max - t_min) * 0.05
    t_min = t_min - t_pad
    t_max = t_max + t_pad

    f_min = np.min(f)
    f_max = np.max(f)
    f_pad = (f_max - f_min) * 0.05
    f_min = f_min - f_pad
    f_max = f_max + f_pad

    if t_min != t_max:
        ax.set_xlim(t_min, t_max)
        ax.set_xticks(t)
        ax.grid(True)
    if f_min != f_max:

        lower_limit = 0.08  # MAX_Y_LIM * .33
        if np.max(f) > lower_limit:
            ax.set_yticks(np.arange(0, MAX_Y_LIM * 1.2, 0.01))
            ax.set_ylim(0, MAX_Y_LIM * 1.1)  # f_min, f_max)
        else:
            ax.set_yticks(np.arange(0, lower_limit * 1.1, 0.01))
            ax.set_ylim(0, lower_limit)  # f_min, f_max)

    line.set_data(t, f)


def interactive_mi_grid(mi_grid, crime_grid, is_conditional_mi=False):
    """
    crime_grid: crime counts N,C,H,W where N time steps, C crime counts
    mi_grid: grid with shape 1,K,H,W where K is the max number of time offset
    """
    _, _, n_rows, n_cols = mi_grid.shape

    fig = plt.figure(figsize=(9, 8))  # , constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    _img0 = ax0.imshow(mi_grid.mean(axis=(0, 1)))
    _img1 = ax1.imshow(crime_grid.mean(axis=(0, 1)))
    line, = ax2.plot([0], [0])

    ax1.set_title("Mean Crime Count")
    if is_conditional_mi:
        ax2_title = "CMI per Temporal Offset"
        ax0.set_title("Conditional Mutual Information (CMI) Mean over Offset")
        ax2.set_title(ax2_title)
        ax2.set_ylabel("CMI - $I(C_{t},C_{t-k}|DoW_{t},DoW_{t-k})$")  # give I(C)
        ax2.set_xlabel("Offset in Days (k)")
    else:
        ax0.set_title("Mutual Information (MI) Mean over Offset")
        ax1.set_title("Crime Rate Grid")
        ax2_title = "MI per Temporal Offset"
        ax2.set_title(ax2_title)
        ax2.set_ylabel("MI - $I(C_{t},C_{t-k})$")  # give I(C)
        ax2.set_xlabel("Offset in Days (k)")

    def draw(row_ind, col_ind):
        ax2.set_title(f"{ax2_title} - {col_ind, row_ind}")

        f = mi_grid[0, :, row_ind, col_ind]
        t = np.arange(1, len(f) + 1)  # start at one because offset starts at 1

        t_min = np.min(t)
        t_max = np.max(t)
        t_pad = (t_max - t_min) * 0.05
        t_min = t_min - t_pad
        t_max = t_max + t_pad

        f_min = np.min(f)
        f_max = np.max(f)
        f_pad = (f_max - f_min) * 0.05
        f_min = f_min - f_pad
        f_max = f_max + f_pad

        if t_min != t_max:
            ax2.set_xlim(t_min, t_max)
            ax2.set_xticks(t)
            ax2.grid(True)
        if f_min != f_max:
            ax2.set_ylim(f_min, f_max)

        line.set_data(t, f)
        fig.canvas.draw()

    def on_click(event):
        log.info(f"event => {pformat(event.__dict__)}")
        if hasattr(event, "xdata") and hasattr(event, "ydata"):
            if event.xdata and event.ydata:  # check that axis is the imshow
                row_ind = int(np.round(event.ydata))
                col_ind = int(np.round(event.xdata))
                draw(row_ind, col_ind)
        return True

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()


def interactive_mi_two_plots(mi_grid, cmi_grid, crime_grid):
    """
    crime_grid: crime counts N,C,H,W where N time steps, C crime counts
    mi_grid: grid with shape 1,K,H,W where K is the max number of time offset
    cmi_grid: grid with shape 1,K,H,W where K is the max number of time offset
    """
    _, _, n_rows, n_cols = mi_grid.shape

    fig = plt.figure(figsize=(9, 8))  # , constrained_layout=True)

    gs = fig.add_gridspec(11, 6)
    ax_mi_img = fig.add_subplot(gs[0:3, 0:2])
    ax_cmi_img = fig.add_subplot(gs[4:7, 0:2])
    ax_crime_img = fig.add_subplot(gs[8:11, 0:2])

    ax_mi_curve = fig.add_subplot(gs[0:5, 2:])
    ax_cmi_curve = fig.add_subplot(gs[6:11, 2:])

    _img_mi = ax_mi_img.imshow(mi_grid.mean(axis=(0, 1)))
    _img_cmi = ax_cmi_img.imshow(cmi_grid.mean(axis=(0, 1)))
    _img_crime = ax_crime_img.imshow(crime_grid.mean(axis=(0, 1)))
    line_mi, = ax_mi_curve.plot([0], [0])
    line_cmi, = ax_cmi_curve.plot([0], [0])

    ax_crime_img.set_title("Mean Crime Count")
    ax_crime_img.set_title("Crime Rate Grid")

    ax_mi_curve_title = "MI per Temporal Offset"
    ax_mi_img.set_title("Mutual Information (MI)\nMean over Time Offset")
    ax_mi_curve.set_title(ax_mi_curve_title)
    ax_mi_curve.set_ylabel("MI - $I(C_{t},C_{t-k})$")  # give I(C)
    ax_mi_curve.set_xlabel("Offset in Days (k)")

    ax_cmi_curve_title = "CMI per Temporal Offset"
    ax_cmi_img.set_title("Conditional Mutual Information (CMI)\nMean over Time Offset")
    ax_cmi_curve.set_title(ax_cmi_curve_title)
    ax_cmi_curve.set_ylabel("CMI - $I(C_{t},C_{t-k}|DoW_{t},DoW_{t-k})$")  # give I(C)
    ax_cmi_curve.set_xlabel("Offset in Days (k)")

    def draw(row_ind, col_ind):
        ax_mi_curve.set_title(f"{ax_mi_curve_title} for {col_ind, row_ind}")
        ax_cmi_curve.set_title(f"{ax_cmi_curve_title} for {col_ind, row_ind}")

        f_mi = mi_grid[0, :, row_ind, col_ind]
        f_cmi = cmi_grid[0, :, row_ind, col_ind]
        t = np.arange(1, len(f_mi) + 1)  # start at one because offset starts at 1

        set_line(f=f_mi, t=t, ax=ax_mi_curve, line=line_mi)
        set_line(f=f_cmi, t=t, ax=ax_cmi_curve, line=line_cmi)
        fig.canvas.draw()

    def on_click(event):
        # log.info(f"event => {pformat(event.__dict__)}")
        if hasattr(event, "xdata") and hasattr(event, "ydata"):
            if event.xdata and event.ydata:  # check that axis is the imshow
                row_ind = int(np.round(event.ydata))
                col_ind = int(np.round(event.xdata))
                draw(row_ind, col_ind)
        return True

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.subplots_adjust(
        left=0.,
        right=0.9,
        bottom=0.1,
        top=0.95,
        wspace=0.3,
        hspace=0.1,
    )

    plt.tight_layout()
    plt.show()


def interactive_mi_one_plot(mi_grid, cmi_grid, crime_grid):
    """
    crime_grid: crime counts N,C,H,W where N time steps, C crime counts
    mi_grid: grid with shape 1,K,H,W where K is the max number of time offset
    cmi_grid: grid with shape 1,K,H,W where K is the max number of time offset
    """
    _, _, n_rows, n_cols = mi_grid.shape

    fig = plt.figure(figsize=(9, 8))  # , constrained_layout=True)

    gs = fig.add_gridspec(2, 3)
    ax_crime_img = fig.add_subplot(gs[0, 0])
    ax_mi_img = fig.add_subplot(gs[0, 1])
    ax_cmi_img = fig.add_subplot(gs[0, 2])
    ax_mi_curve = fig.add_subplot(gs[1, :])

    _img_mi = ax_mi_img.imshow(mi_grid.mean(axis=(0, 1)))
    _img_cmi = ax_cmi_img.imshow(cmi_grid.mean(axis=(0, 1)))
    _img_crime = ax_crime_img.imshow(crime_grid.mean(axis=(0, 1)))
    line_mi, = ax_mi_curve.plot([0], [0], label="$I(C_{t};C_{t-k})$")
    line_cmi, = ax_mi_curve.plot([0], [0], label="$I(C_{t};C_{t-k}|DoW_{t},DoW_{t-k})$")

    ax_crime_img.set_title("Mean Crime Count")
    ax_crime_img.set_title("Crime Rate Grid")

    global normalize
    if normalize:
        ax_mi_curve_title = "Mutual Information per Temporal Offset (Normalized)"
    else:
        ax_mi_curve_title = "Mutual Information per Temporal Offset"

    ax_mi_img.set_title("Mutual Information (MI)\nMean over Time Offset")
    ax_cmi_img.set_title("Conditional Mutual Information (CMI)\nMean over Time Offset")
    ax_mi_curve.set_title(ax_mi_curve_title)
    ax_mi_curve.set_ylabel("Mutual Information (bits)")  # give I(C)
    ax_mi_curve.set_xlabel("Offset in Days (k)")
    plt.legend()

    def draw(row_ind, col_ind):
        ax_mi_curve.set_title(f"{ax_mi_curve_title} for {col_ind, row_ind}")

        f_mi = mi_grid[0, :, row_ind, col_ind]
        f_cmi = cmi_grid[0, :, row_ind, col_ind]
        t = np.arange(1, len(f_mi) + 1)  # start at one because offset starts at 1

        set_line(f=f_mi, t=t, ax=ax_mi_curve, line=line_mi)
        set_line(f=f_cmi, t=t, ax=ax_mi_curve, line=line_cmi)
        fig.canvas.draw()

    def on_click(event):
        # log.info(f"event => {pformat(event.__dict__)}")
        if hasattr(event, "xdata") and hasattr(event, "ydata"):
            if event.xdata and event.ydata:  # check that axis is the imshow
                row_ind = int(np.round(event.ydata))
                col_ind = int(np.round(event.xdata))
                draw(row_ind, col_ind)
        return True

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.subplots_adjust(
        left=0.,
        right=0.9,
        bottom=0.1,
        top=0.95,
        wspace=0.3,
        hspace=0.1,
    )

    plt.tight_layout()
    plt.show()


def main():
    conf, shaper, sparse_crimes = setup(data_sub_path="T24H-X850M-Y880M_2012-01-01_2019-01-01")
    # conf, shaper, sparse_crimes = setup(data_sub_path="T24H-X425M-Y440M_2013-01-01_2017-01-01")

    squeezed_crimes = shaper.squeeze(sparse_crimes)
    # squeezed_crimes[squeezed_crimes > 0] = 1
    squeezed_crimes[squeezed_crimes > 40] = 40
    squeezed_crimes = np.round(np.log2(1 + squeezed_crimes))
    _info(f"squeezed_crimes values =>{np.unique(squeezed_crimes)}")

    n, c, l = squeezed_crimes.shape

    # setup the day of the week variables
    dow_n = np.arange(n) % 7
    dow_nc = np.expand_dims(dow_n, (1, 2))
    dow_ncl = np.ones((n, c, l)) * dow_nc

    data = np.concatenate([squeezed_crimes, dow_ncl], axis=1)

    K = 90 # 7 * 6
    file_name = f"arrays_K{K:02d}"
    file_dir = f"{conf.data_path}mutual_info"
    file_location = f"{file_dir}/{file_name}.npy.npz"
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    if os.path.exists(file_location):
        _info(f"Loading existing mutual information data from: {file_dir}")
        with np.load(file_location) as zip_file:  # context helper ensures zip_file is closed
            mi_arr = zip_file["mi_arr"]
            cmi_arr = zip_file["cmi_arr"]
    else:
        _info(f"Mutual information does not exist at: {file_dir}")
        _info(f"Generating mutual information")
        # todo abstract into function - persist under data_sub_path
        rv_names = ['RV0_Ct', 'RV0_Dt', 'RV1_Ct-k', 'RV1_Dt-k']  # Ct: crime at t. Dt: day of week at t
        mi_list = []
        cmi_list = []
        for i in range(l):
            if i % (l // 10) == 0:
                print(f"{i + 1}/{l} => {(i + 1) / l * 100}%")
            mi_list.append([])
            cmi_list.append([])
            for k in range(0, K + 1):  # K is the maximum
                if k == 0:
                    joint = np.concatenate([data[:, :, i], data[:, :, i]], axis=1)
                else:
                    joint = np.concatenate([data[k:, :, i], data[:-k, :, i]], axis=1)
                val, cnt = np.unique(joint, return_counts=True, axis=0)
                prb = cnt / np.sum(cnt)
                table = {}
                for k_, v_ in zip(list(map(tuple, val)), list(prb)):
                    table[k_] = v_

                # self_information is just the entropy of the variables
                rv = SparseDiscreteTable(rv_names=rv_names, table=table)
                if k == 0:
                    # entropy of the variable we are trying to measure - minum bytes needed to encode the distribution
                    self_information = rv['RV0_Ct',].entropy()
                    cmi_list[i].append(self_information)
                    mi_list[i].append(self_information)
                else:
                    mi = rv.mutual_information(rv_names_0=['RV0_Ct'],
                                               rv_names_1=['RV1_Ct-k'])
                    cmi = rv.conditional_mutual_information(rv_names_0=['RV0_Ct'],
                                                            rv_names_1=['RV1_Ct-k'],
                                                            rv_names_condition=['RV0_Dt', 'RV1_Dt-k'])
                    cmi_list[i].append(cmi)
                    mi_list[i].append(mi)

        mi_arr = np.array(mi_list)
        cmi_arr = np.array(cmi_list)
        np.savez_compressed(file=file_location,
                            cmi_arr=cmi_arr,
                            mi_arr=mi_arr)
        _info(f"Saved mutual information arrays at: {file_location}")

    global normalize
    normalize = False
    mi_grid = construct_mi_grid(mi_arr=mi_arr, shaper=shaper, normalize=normalize)
    cmi_grid = construct_mi_grid(mi_arr=cmi_arr, shaper=shaper, normalize=normalize)

    global MAX_Y_LIM
    MAX_Y_LIM = max(mi_grid.max(), cmi_grid.max())
    _info(f"mi_grid.max(), cmi_grid.max() => {mi_grid.max(), cmi_grid.max()}")
    _info(f"MAX_Y_LIM => {MAX_Y_LIM}")

    interactive_mi_one_plot(mi_grid=mi_grid,
                            cmi_grid=cmi_grid,
                            crime_grid=sparse_crimes)

    # interactive_mi_two_plots(mi_grid=mi_grid,
    #                          cmi_grid=cmi_grid,
    #                          crime_grid=sparse_crimes)

    # interactive_mi_grid(mi_grid=mi_grid, crime_grid=sparse_crimes, is_conditional_mi=False)
    # interactive_mi_grid(mi_grid=cmi_grid, crime_grid=sparse_crimes,is_conditional_mi=True)


if __name__ == '__main__':
    main()
