

def interactive_mi_grid(mi_grid, crime_grid, is_conditional_mi=False):
    """
    crime_grid: crime counts N,C,H,W where N time steps, C crime counts
    mi_grid: grid with shape 1,K,H,W where K is the max number of time offset
    """

    _,_,n_rows,n_cols = mi_grid.shape

    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])


    img0 = ax0.imshow(mi_grid.mean(axis=(0,1)))
    img1 = ax1.imshow(crime_grid.mean(axis=(0,1)))
    line, = ax2.plot([0], [0])

    ax1.set_title("Mean Crime Count")
    if is_conditional_mi:
        ax0.set_title("Conditional Mutual Information (CMI) Mean over Offset")
        ax2.set_title("CMI per Temporal Offset")
        ax2.set_ylabel("CMI - $I(C_{t},C_{t-k}|DoW_{t},DoW_{t-k})$") # give I(C)
        ax2.set_xlabel("Offset in Days (k)")
    else:
        ax0.set_title("Mutual Information (MI) Mean over Offset")
        ax1.set_title("Crime Rate Grid")
        ax2.set_title("MI per Temporal Offset")
        ax2.set_ylabel("MI - $I(C_{t},C_{t-k})$") # give I(C)
        ax2.set_xlabel("Offset in Days (k)")

    def draw(row_ind, col_ind):
        f = mi_grid[0,:,row_ind, col_ind]
        t = np.arange(1,len(f)+1) # start at one because offset starts at 1

        t_min = t.min()
        t_max = t.max()
        t_pad = (t_max - t_min) * 0.05
        t_min = t_min - t_pad
        t_max = t_max + t_pad

        f_min = f.min()
        f_max = f.max()
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

    def on_click(event):
        print(f"event => {pformat(event.__dict__)}")
        if hasattr(event, "xdata") and hasattr(event, "ydata"):
            if event.xdata and event.ydata:  # check that axis is the imshow
                row_ind = int(np.round(event.ydata))
                col_ind = int(np.round(event.xdata))
                draw(row_ind, col_ind)
        return True

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()