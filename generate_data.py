import os
import pickle
from pprint import pprint

from matplotlib import rcParams
import matplotlib.pyplot as plt
from utils.utils import *
from utils.data_processing import *
import logging as log
from logger.logger import setup_logging
from utils.preprocessing import Shaper
from sys import argv
from utils.data_processing import encode_category

if __name__ == "__main__":
    if len(argv) < 2:
        raise FileNotFoundError("Please specify config hex reference as argument")
    else:
        ref = argv[1]
        config_file_name = f"./config/generate_data_{ref}.json"

    setup_logging(save_dir="./logs/", file_name="generate_data.log")
    log.info("=====================================BEGIN=====================================")
    # set the plotting format
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.family'] = 'STIXGeneral'

    # set all values from the config
    config = read_json(config_file_name)
    dT = config["dT"]

    x_scale = config["x_scale"]
    y_scale = config["y_scale"]

    # used as info file for the folder where the data is created
    meta_info = dict()
    meta_info["createdOn"] = datetime.now().strftime("%Y%m%dT%H%M")
    load_folder_raw = "./data/raw/"
    load_folder_original = "./data/original/"

    ######################################################
    #                     LOAD DATA                      #
    ######################################################
    # external information (demographic and weather)
    # not these are only weather values for 2014
    weather_df = pd.read_pickle(load_folder_raw + "weather_minmax_normed.pkl")
    weather_df.index = weather_df.date  # set datetime index todo: move to data-processing
    weather_df.drop(columns="date", inplace=True)  # drop the time column
    extra_day = weather_df.iloc[-1]
    extra_day.name = extra_day.name + pd.DateOffset(1)
    weather_df = weather_df.append(extra_day)
    # re-sample weather data
    is_gte_24hours = 'D' in dT or dT == '24H'
    if is_gte_24hours:
        weather_df = weather_df.resample(dT).mean()
    else:
        weather_df = weather_df.resample(dT).pad()
    weather_df = weather_df.iloc[:-1]
    weather_vectors = weather_df.values

    # physical coords and their tracts
    valid_points = np.load(load_folder_raw + "valid_points.npy")
    valid_tracts = np.load(load_folder_raw + "valid_tracts.npy")
    invalid_tracts = np.load(load_folder_raw + "invalid_tracts.npy")
    tract2points = pd.read_pickle(load_folder_raw + "tract2points.pkl")
    point2tract = pd.read_pickle(load_folder_raw + "point2tract.pkl")
    tract_boundaries = pd.read_pickle(load_folder_raw + "tract_boundaries.pkl")
    tract2km2 = pd.read_pickle(load_folder_raw + "tract2km2.pkl")

    # link between points and demographic data
    point2spatial_info = pd.read_pickle(load_folder_raw + "point2spatial_info.pkl")  # todo is this normalised?

    ######################################################
    #                PREPROCESS DATA                     #
    ######################################################
    # Time dimensions
    meta_info["dT"] = dT
    # Spatial dimensions

    meta_info["ref"] = ref

    # [20,16] - [10,8] - [5,4] - [3,2] - [1,1]
    xy_scale = np.array([x_scale, y_scale])  # must be integer so that we can easily sample demographic data
    dx, dy = xy_scale * np.array([0.001, 0.001])
    meta_info["x_scale"] = x_scale
    meta_info["y_scale"] = y_scale
    meta_info["dx"] = float(dx)
    meta_info["dy"] = float(dy)
    meta_info["x in metres"] = 85000 * float(dx)
    meta_info["y in metres"] = 110000 * float(dy)

    log.info("Cell sizes: %.3f m in x direction and %.3f m in y direction" % (85000 * dx, 110000 * dy))

    # crimes = pd.read_pickle(load_folder_raw + "crimes_2012_to_2018.pkl")
    crimes = pd.read_pickle(load_folder_raw + "crimes_2012_to_2018_new.pkl")

    # CHOOSE CRIME TYPES
    valid_crime_types = config["crime_types"]
    # valid_crime_types = [
    #     "THEFT",
    #     "BATTERY",
    #     "CRIMINAL DAMAGE",
    #     "NARCOTICS",
    #     "ASSAULT",
    #     "BURGLARY",
    #     "MOTOR VEHICLE THEFT",
    #     "ROBBERY",
    # ]

    # filter useless crime types
    crimes = crimes[crimes["Primary Type"].isin(valid_crime_types)]

    # take out western most tract to simplify things for the cnn
    crimes = crimes[crimes.tract != 7706.02]

    start_date = config["start_date"]
    end_date = config["end_date"]
    meta_info["start_date"] = start_date
    meta_info["end_date"] = end_date

    t_range = pd.date_range(start_date, end_date, freq=dT)
    crimes = crimes[crimes.Date < t_range[-1]]  # choose only crimes which lie in the valid time range
    crimes = crimes[crimes.Date >= t_range[0]]  # choose only crimes which lie in the valid time range
    # crimes.Date = crimes.Date.dt.floor(dT) # DON"T ROUND,
    # ONLY FLOOR, OTHERWISE THE WEATHER DATA DOESNT LINE UP PROPERLY.

    t_min = pd.Series(crimes.Date.min()).astype(np.int64)[0]
    dt = pd.Series(t_range[1] - t_range[0]).astype(np.int64)[0]

    t = crimes.Date.astype(np.int64)
    t = t - t_min
    t = t // dt
    crimes["t"] = t

    x_max_valid, y_max_valid = valid_points.max(0)
    x_min_valid, y_min_valid = valid_points.min(0)

    lat_max = np.round(config["lat_max"] / dy) * dy
    lat_min = np.round(config["lat_min"] / dy) * dy
    lon_max = np.round(config["lon_max"] / dx) * dx
    lon_min = np.round(config["lon_min"] / dx) * dx

    x_min_valid = max(x_min_valid, lon_min)
    y_min_valid = max(y_min_valid, lat_min)
    x_max_valid = min(x_max_valid, lon_max)
    y_max_valid = min(y_max_valid, lat_max)

    meta_info["x_min_valid"] = x_min_valid
    meta_info["y_min_valid"] = y_min_valid
    meta_info["x_max_valid"] = x_max_valid
    meta_info["y_max_valid"] = y_max_valid

    # we know all crimes have defined demographics
    # spatial discritization with step
    crimes["X"] = crimes.Longitude
    crimes["Y"] = crimes.Latitude

    # round Long and Lat to nearest increment of dx and dy
    # all spots are defined?

    crimes.X = np.round(crimes.X / dx) * dx
    crimes.Y = np.round(crimes.Y / dy) * dy

    crimes.X = np.round(crimes.X, decimals=3)  # used to make sure we can still hash coords
    crimes.Y = np.round(crimes.Y, decimals=3)  # rounding ensures floating point issues are dealt with

    log.info(f"Number of total crimes: {len(crimes)}")
    crimes = crimes[
        (crimes.X <= x_max_valid) & (crimes.X >= x_min_valid) & (crimes.Y >= y_min_valid) & (crimes.Y <= y_max_valid)]
    log.info(f"Number of crimes in valid spatial range: {len(crimes)}")

    # x_range = np.arange(crimes.X.min(),crimes.X.max()+dx,dx)
    # .001 because that is our smallest element dx can be bigger
    # y_range = np.arange(crimes.Y.min(),crimes.Y.max()+dy,dy)
    x_min, x_max = crimes.X.min(), crimes.X.max()
    y_min, y_max = crimes.Y.min(), crimes.Y.max()

    # .001 because that is our smallest element dx can be bigger
    x_range = np.arange(crimes.X.min(), np.round(crimes.X.max() + dx, decimals=3), dx)
    y_range = np.arange(crimes.Y.min(), np.round(crimes.Y.max() + dy, decimals=3), dy)

    x_range = x_range.round(decimals=3)  # used to make sure we can still hash coords
    y_range = y_range.round(decimals=3)

    # filter because of floating point issues
    x_range = x_range[(x_range >= x_min) & (x_range <= x_max)]
    y_range = y_range[(y_range >= y_min) & (y_range <= y_max)]

    x_min, x_max = x_range[0], x_range[-1]
    y_min, y_max = y_range[0], y_range[-1]

    crimes["x"] = np.array(np.round((crimes.X - x_min) / dx), dtype=int)
    crimes["y"] = np.array(np.round((crimes.Y - y_min) / dy), dtype=int)

    # only take crimes that land on nodes that have demographic info
    all_crime_spots = crimes[["X", "Y"]].values
    crimes["xy"] = list(map(tuple, all_crime_spots))
    valid_crime_spots = set2d(crimes[["X", "Y"]].values) - (set2d(crimes[["X", "Y"]].values) - set2d(valid_points))
    drop_crime_spots = list(set2d(all_crime_spots) - valid_crime_spots)
    valid_crime_spots = np.array(list(valid_crime_spots))

    indices_to_drop = []

    for i in range(len(drop_crime_spots)):
        log.info(f"dropped {i}")
        indices = crimes[crimes.xy == drop_crime_spots[i]].index
        crimes.drop(index=indices, inplace=True)

    log.info(f"Number of crimes valid spatial range and on nodes with demographic info: {len(crimes)}")

    X, Y = np.meshgrid(x_range, y_range)
    # crimes["Primary Type"].value_counts()
    t_size = len(t_range) - 1  # dates are an extra one - the range indicates start and end walls of each cell
    x_size = len(x_range)  # x_range are the means of each cell
    y_size = len(y_range)  # y_range are the means of each cell

    log.info(f"t_size:\t{t_size}\nx_size:\t{x_size}\ny_size:\t{y_size}")

    A = crimes[["t", "x", "y"]].values[:]

    log.info(f"A.shape -> {A.shape}")
    log.info(f"t_size, x_size, y_size -> {t_size}, {x_size}, {y_size}")
    log.info(f"crimes.t.max(), crimes.x.max(), crimes.y.max() -> {crimes.t.max()}, {crimes.x.max()},{crimes.y.max()}")
    log.info(f"t_range[-1] -> {t_range[-1]}")
    log.info(f"t_range[0] -> {t_range[0]}")

    meta_info["t_size"] = t_size
    meta_info["x_size"] = x_size
    meta_info["y_size"] = y_size

    meta_info["crimes.t.max()"] = int(crimes.t.max())
    meta_info["crimes.t.min()"] = int(crimes.t.min())
    meta_info["crimes.x.max()"] = int(crimes.x.max())
    meta_info["crimes.x.min()"] = int(crimes.x.min())
    meta_info["crimes.y.max()"] = int(crimes.y.max())
    meta_info["crimes.y.min()"] = int(crimes.y.min())

    crime_grids = make_grid(A, t_size, x_size, y_size)

    ######################################################
    #          TRACTS GRIDS DATA GENERATION               #
    ######################################################
    # Creates Grid where the cell value is the total crime in that tract for that time step
    tract2index = {}
    for i, tr in enumerate(valid_tracts):
        tract2index[tr] = i

    trindex = []
    for tr in crimes.tract.values:
        trindex.append(tract2index[tr])

    crimes['trindex'] = trindex
    tracts = np.zeros((t_size, len(valid_tracts)))  # crime count in tracts over time - can be used for lstm

    tract_info = crimes[['t', 'trindex']].values

    for t, tr in tract_info:
        tracts[t, tr] += 1

    # make grid by the number of crimes in that tract
    tract_count_grids = np.zeros(crime_grids.shape)

    # for x, y in valid_crime_spots:  # leads to some missing data
    for x, y in valid_points:
        tr = point2tract[x, y]
        if x in x_range and y in y_range:
            x = np.argwhere(x_range == x)[0, 0]
            y = np.argwhere(y_range == y)[0, 0]
            info_ = tracts[:, tract2index[tr]]
            tract_count_grids[:, y_size - y - 1, x] = np.array(info_)

    # Adding any crime related data to the channels, e.g. tract counts if we want to
    tract_count_grids = np.expand_dims(tract_count_grids, axis=1)
    """
    note: tract_count_grids has values where the shaper loses values - that is why
    when un-squeezing the tract_count_grids using the shaper the grid counts do not match up  
    """

    ######################################################
    #          DEMOGRAPHIC DATA GENERATION               #
    ######################################################
    # todo implement interpolation for this as well.

    num_demog_feats = 37
    demog_grid = np.zeros((num_demog_feats, y_size, x_size))  # grids with demographic info

    err = []
    for x in range(x_size):
        for y in range(y_size):
            X_ = x_range[x]
            Y_ = y_range[y]
            try:
                demog_grid[:, y_size - y - 1, x] = np.array(
                    point2spatial_info[X_, Y_])  # should be redone with filtered census data
            except KeyError:
                err.append((X_, Y_))

    log.info(f"sum(demog_grid):\t {demog_grid.sum()}")
    log.info(f"x_size*y_size:\t\t {x_size * y_size}")
    log.info(f"len(err):\t\t {len(err)}")

    # street view vectors
    # some cells do not have coordinates
    # we use pca to compress the feature vector (pca not necessarily needed)
    # knn to fill in featureless cell with the values of the closest feature full cell
    point2feats_res18 = pd.read_pickle(load_folder_raw + "point2feats_res18.pkl")

    from sklearn.decomposition import PCA

    coords = np.array(list(point2feats_res18.keys()))
    feats = np.array(list(point2feats_res18.values()))

    n_components = 512

    pca = PCA(n_components=n_components)
    c = pca.fit_transform(feats)
    c = (c - c.min()) / (c.max() - c.min())

    from sklearn.neighbors import KNeighborsRegressor

    knn = KNeighborsRegressor(n_neighbors=1, weights="uniform")
    knn.fit(coords, c)
    valid_feats = knn.predict(valid_points)

    point2feats = dict()
    for i, (x, y) in enumerate(valid_points):
        point2feats[x, y] = valid_feats[i]

    street_grid = np.zeros((n_components, y_size, x_size))  # grids with demographic info
    err = []
    for x in range(x_size):
        for y in range(y_size):
            X_ = x_range[x]
            Y_ = y_range[y]
            try:
                street_grid[:, y_size - y - 1, x] = np.array(
                    point2feats[X_, Y_])  # should be redone with filtered census data
            except KeyError:
                err.append((X_, Y_))

    log.info(f"sum(street_grid):\t {street_grid.sum()}")
    log.info(f"x_size*y_size:\t\t {x_size * y_size}")
    log.info(f"len(err):\t\t {len(err)}")

    #########################################################################
    #                           CRIME TYPES GRID                            #
    #########################################################################

    c2i = {name: i for i, name in enumerate(valid_crime_types)}


    # c2i = {
    #     "THEFT": 0,
    #     "BATTERY": 1,
    #     "CRIMINAL DAMAGE": 2,
    #     "NARCOTICS": 3,
    #     "ASSAULT": 4,
    #     "BURGLARY": 5,
    #     "MOTOR VEHICLE THEFT": 6,
    #     "ROBBERY": 7,
    # }  # can also change the values to group certain crimes into a class

    #                       - like battery and assault into a type and theft and robbery and
    #                       motor vehicle theft into a type and narcotics into another type.

    # OTHER OPTIONS
    # c2i = {"THEFT":0,
    # "BATTERY":1,
    # "CRIMINAL DAMAGE":2,
    # "NARCOTICS":3,
    # "ASSAULT":1,
    # "BURGLARY":0,
    # "MOTOR VEHICLE THEFT":0,
    # "ROBBERY":0} # can

    # OTHER OPTIONS
    # c2i = {"THEFT":0,
    # "BATTERY":1,
    # "NARCOTICS":2}

    def type2index(crime_type):
        return c2i.get(crime_type, -1) # all other types are translated to -1

    crimes["c"] = crimes["Primary Type"].apply(type2index)
    # crimes["c"] = encode_category(series=df['Primary Type'],categories=valid_crime_types)

    # FILTER OUT INVALID CRIME TYPES
    crimes = crimes[crimes.c > -1]

    # ONE HOT ENCODING FOR THE CRIME TYPES
    ohe = np.zeros((len(crimes), len(valid_crime_types)), dtype=int)
    for i, c in enumerate(crimes.c):
        ohe[i, c] = 1

    for i, k in enumerate(c2i):
        crimes[k] = ohe[:, i]

    crimes["TOTAL"] = np.ones(len(crimes), dtype=int)

    # INCLUDE ARRESTS
    crimes["Arrest"] = crimes["Arrest"] * 1  # casting bool to int

    # crime_feature_indices = {
    #     0: "TOTAL",
    #     1: "THEFT",
    #     2: "BATTERY",
    #     3: "CRIMINAL DAMAGE",
    #     4: "NARCOTICS",
    #     5: "ASSAULT",
    #     6: "BURGLARY",
    #     7: "MOTOR VEHICLE THEFT",
    #     8: "ROBBERY",
    #     9: "Arrest"
    # }

    crime_feature_indices = ["TOTAL", *valid_crime_types, "Arrest"]

    A = crimes[["t", "x", "y", "TOTAL", *valid_crime_types, "Arrest"]].values
    # A = crimes[["t","b","TOTAL","THEFT", "BATTERY", "NARCOTICS","Arrest"]].values # is used when x and y are flattened

    crime_type_grids = np.zeros((t_size, A.shape[-1] - 3, y_size, x_size))
    # todo convert to sparse data matrices
    for a in A:
        crime_type_grids[a[0], :, y_size - 1 - a[2], a[1]] += a[3:]

    #########################################################################
    #                            SAVE DATA                                  #
    #########################################################################
    # TODO: add the x and y limits to the folder id as well
    save_folder = f"./data/processed/T{dT}-X{int(meta_info['x in metres'])}M-Y{int(meta_info['y in metres'])}M_{meta_info['start_date']}_{meta_info['end_date']}_#{meta_info['ref']}/"

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder + "plots", exist_ok=True)

    # save figures
    figsize = (8, 11)

    v, c = np.unique(crime_grids.flatten(), return_counts=True)
    c = 100 * c / np.sum(c)
    plt.figure(figsize=figsize)
    plt.bar(v, c)
    plt.yticks(np.arange(21) * 5)
    plt.title("Maximum Crimes per Time Step")
    plt.ylabel("Frequency (%)")
    plt.xlabel("Total Crimes per Time Step per Cell")
    plt.grid(True)
    plt.savefig(save_folder + "plots/" + "crime_distribution.png")

    plt.figure(figsize=figsize)
    plt.scatter(X, Y, marker=".", label="Ranged")
    plt.scatter(valid_points[:, 0], valid_points[:, 1], marker=".", label="Valid", s=1)
    plt.scatter(crimes.X, crimes.Y, marker=".", label="Rounded")
    plt.legend(loc=3, prop={"size": 20})
    plt.ylabel("Latitude")
    plt.xlabel("Longitude")
    plt.savefig(save_folder + "plots/" + "scatter_map.png")

    plt.figure(figsize=figsize)
    plt.title("Maximum Crimes per Time Step")
    plt.imshow(crime_grids.max(0), cmap="viridis")
    plt.colorbar()
    plt.ylabel("Y Coordinate")
    plt.xlabel("X Coordinate")
    plt.savefig(save_folder + "plots/" + "crimes_max.png")

    plt.figure(figsize=figsize)
    plt.title("Mean Crimes per Time Step")
    plt.imshow(crime_grids.mean(0), cmap="viridis")
    plt.colorbar()
    plt.ylabel("Y Coordinate")
    plt.xlabel("X Coordinate")
    plt.savefig(save_folder + "plots/" + "crimes_mean.png")

    plt.figure(figsize=figsize)
    plt.title("Demographics Max Value")
    plt.imshow(demog_grid.max(0), cmap="viridis")
    plt.ylabel("Y Coordinate")
    plt.xlabel("X Coordinate")
    plt.savefig(save_folder + "plots/" + "demographics_max.png")

    plt.figure(figsize=figsize)
    plt.title("Street View Data Max Value")
    plt.ylabel("Y Coordinate")
    plt.xlabel("X Coordinate")
    plt.imshow(street_grid.max(0), cmap="viridis")
    plt.savefig(save_folder + "plots/" + "street_grid_max.png")

    #  ENSURE DIMS (N, C, H, W) FORMAT
    crime_grids = np.expand_dims(crime_grids, axis=1)
    demog_grid = np.expand_dims(demog_grid, axis=0)
    street_grid = np.expand_dims(street_grid, axis=0)

    # #  compress using shaper
    # shaper = Shaper(crime_grids)
    # crime_type_grids = shaper.squeeze(crime_type_grids)
    # crime_grids = shaper.squeeze(crime_grids)
    # last_year_crime_grids = shaper.squeeze(last_year_crime_grids)
    # tract_count_grids = shaper.squeeze(tract_count_grids)
    # demog_grid = shaper.squeeze(demog_grid)
    # street_grid = shaper.squeeze(street_grid)
    # with open(f"{save_folder}shaper.pkl", "wb") as shaper_file:
    #     pickle.dump(shaper, shaper_file)

    # save generated data
    # TODO ENSURE ALL SPATIAL DATA IS IN FORM N, C, H, W -> EVEN IF C = 1 SHOULD BE N, 1, H, W
    for g in [crime_type_grids, crime_grids, demog_grid, street_grid]:
        assert len(g.shape) == 4

    # note - we only normalise later as some models use different normalisation techniques
    time_vectors = encode_time_vectors(t_range, month_divisions=10, year_divisions=10, kind='ohe')  # kind='sincos')

    np.savez_compressed(save_folder + "generated_data.npz",
                        crime_feature_indices=crime_feature_indices,
                        crime_types_grids=crime_type_grids,  # sum crimes, crime types, arrests
                        crime_grids=crime_grids,  # sum crimes only
                        tract_count_grids=tract_count_grids,  # sum crimes of tracts
                        demog_grid=demog_grid,
                        street_grid=street_grid,
                        time_vectors=time_vectors,
                        weather_vectors=weather_vectors,  # will not be using weather data
                        x_range=x_range,
                        y_range=y_range)

    pd.to_pickle(t_range, f"{save_folder}t_range.pkl")
    # way weather dates are too short time_vectors should be more years
    # - open weather data has more data but has quite a few gaps
    # np.save(folder+"weather_vectors.npy", weather_vectors)

    write_json(meta_info, f"{save_folder}info.json")
    write_json(config, f"{save_folder}generate_data_config.json")

    log.info("=====================================END=====================================")
