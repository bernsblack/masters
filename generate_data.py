import os
from matplotlib import rcParams
import matplotlib.pyplot as plt
from utils.utils import *
from utils.data_processing import *

if __name__ == "__main__":
    # set the plotting format
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.family'] = 'STIXGeneral'

    print("GENERATING DATA...")

    # set all values from the config
    config = read_json("./config/generate_data.json")
    dT = config["dT"]
    scale = config["scale"]  # scale the area

    # used as info file for the folder where the data is created
    info = dict()
    info["createdOn"] = datetime.now().strftime("%Y%m%dT%H%M")
    load_folder_raw = "./data/raw/"
    load_folder_original = "./data/original/"

    ######################################################
    #                     LOAD DATA                      #
    ######################################################
    # external information (demographic and weather)
    weather = pd.read_pickle(load_folder_raw + "weather_minmax_normed.pkl")
    # todo is this needed or do we get all info from point2spatial_info
    # census = pd.read_pickle(load_folder + "census_minmax_normed.pkl")

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
    info["dT"] = dT
    # Spatial dimensions
    xy_scale = scale * np.array([10, 8])  # must be integer so that we can easily sample demographic data
    dx, dy = xy_scale * np.array([0.001, 0.001])
    info["dx"] = float(dx)
    info["dy"] = float(dy)
    info["x in metres"] = 85000 * float(dx)
    info["y in metres"] = 110000 * float(dy)

    print("Cell sizes: %.3f m in x direction and %.3f m in y direction" % (85000 * dx, 110000 * dy))

    crimes = pd.read_pickle(load_folder_raw + "crimes_2012_to_2018.pkl")
    # crimes = pd.read_pickle(load_folder_raw + "crimes.pkl")
    # add tracts columns - select only valid tracts, but the valid points should be filtered in any case
    # case to date time type the string?

    # take out western most tract to simplify things for the cnn
    crimes = crimes[crimes.tract != 7706.02]

    start_date = config["start_date"]
    end_date = config["end_date"]
    info["start_date"] = start_date
    info["end_date"] = end_date

    crimes["DateTime"] = crimes.Date

    t_range = pd.date_range(start_date, end_date, freq=dT)
    crimes = crimes[crimes.DateTime <= t_range[-1]]  # choose only crimes which lie in the valid time range
    crimes = crimes[crimes.DateTime >= t_range[0]]  # choose only crimes which lie in the valid time range
    # crimes.DateTime = crimes.DateTime.dt.floor(dT) # DON"T ROUND,
    # ONLY FLOOR, OTHERWISE THE WEATHER DATA DOESNT LINE UP PROPERLY.

    t_min = pd.Series(crimes.DateTime.min()).astype(np.int64)[0]
    dt = pd.Series(t_range[1] - t_range[0]).astype(np.int64)[0]

    t = crimes.DateTime.astype(np.int64)
    t = t - t_min
    t = t // dt
    crimes["t"] = t

    x_max_valid, y_max_valid = valid_points.max(0)
    x_min_valid, y_min_valid = valid_points.min(0)

    # we know all crimes have defined demographics
    # spatial discritization with step
    crimes["X"] = crimes.Longitude
    crimes["Y"] = crimes.Latitude

    # round Long and Lat to nearest increment of dx and dy
    # all spots are defined?

    crimes.X = np.round(crimes.X / dx) * dx
    crimes.Y = np.round(crimes.Y / dy) * dy

    crimes.X = np.round(crimes.X, decimals=3)  # used to make sure we can still hash coords
    crimes.Y = np.round(crimes.Y, decimals=3)

    print("Number of total crimes: ", len(crimes))
    crimes = crimes[
        (crimes.X <= x_max_valid) & (crimes.X >= x_min_valid) & (crimes.Y >= y_min_valid) & (crimes.Y <= y_max_valid)]
    print("Number of crimes in valid spatial range:", len(crimes))

    # x_range = np.arange(crimes.X.min(),crimes.X.max()+dx,dx)
    # .001 because that is our smallest element dx can be bigger
    # y_range = np.arange(crimes.Y.min(),crimes.Y.max()+dy,dy)
    x_min, x_max = crimes.X.min(), crimes.X.max()
    y_min, y_max = crimes.Y.min(), crimes.Y.max()

    x_range = np.arange(crimes.X.min(), np.round(crimes.X.max() + dx, decimals=3),
                        dx)  # .001 because that is our smallest element dx can be bigger
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
        print(f"dropped {i}")
        indices = crimes[crimes.xy == drop_crime_spots[i]].index
        crimes.drop(index=indices, inplace=True)

    print("Number of crimes valid spatial range and on nodes with demographic info:", len(crimes))

    X, Y = np.meshgrid(x_range, y_range)
    # crimes["Primary Type"].value_counts()
    t_size = len(t_range)
    x_size = len(x_range)
    y_size = len(y_range)

    print(f"t_size:\t{t_size}\nx_size:\t{x_size}\ny_size:\t{y_size}")

    # A = crimes[crimes["Primary Type"] == "BURGLARY"][["t","x","y",]].values[:] # crime specific
    A = crimes[["t", "x", "y", ]].values[:]

    print(f"A.shape -> {A.shape}")
    print(f"t_size, x_size, y_size -> {t_size}, {x_size}, {y_size}")
    print(f"crimes.t.max(), crimes.x.max(), crimes.y.max() -> {crimes.t.max()}, {crimes.x.max()},{crimes.y.max()}")
    print(f"t_range[-1] -> {t_range[-1]}")
    print(f"t_range[0] -> {t_range[0]}")

    info["t_size"] = t_size
    info["x_size"] = x_size
    info["y_size"] = y_size

    info["crimes.t.max()"] = int(crimes.t.max())
    info["crimes.t.min()"] = int(crimes.t.min())
    info["crimes.x.max()"] = int(crimes.x.max())
    info["crimes.x.min()"] = int(crimes.x.min())
    info["crimes.y.max()"] = int(crimes.y.max())
    info["crimes.y.min()"] = int(crimes.y.min())

    grids = make_grid(A, t_size, x_size, y_size)

    ######################################################
    #          DEMOGRAPHIC DATA GENERATION               #
    ######################################################
    # todo implement interpolation for this as well.

    num_demog_feats = 37
    demog_grid = np.zeros((num_demog_feats, y_size, x_size))  # grids with demographic info

    err = []
    for x in range(x_size):
        for y in range(y_size):
            X = x_range[x]
            Y = y_range[y]
            try:
                demog_grid[:, y_size - y - 1, x] = np.array(
                    point2spatial_info[X, Y])  # should be redone with filtered census data
            except KeyError:
                err.append((X, Y))

    print("sum(demog_grid):\t", demog_grid.sum())
    print("x_size*y_size:\t\t", x_size * y_size)
    print("len(err):\t\t", len(err))

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
            X = x_range[x]
            Y = y_range[y]
            try:
                street_grid[:, y_size - y - 1, x] = np.array(
                    point2feats[X, Y])  # should be redone with filtered census data
            except KeyError:
                err.append((X, Y))

    print("sum(street_grid):\t", street_grid.sum())
    print("x_size*y_size:\t\t", x_size * y_size)
    print("len(err):\t\t", len(err))

    #########################################################################
    #                           CRIME TYPES GRID                            #
    #########################################################################
    # CHOOSE CRIME TYPES

    c2i = {"THEFT": 0,
           "BATTERY": 1,
           "CRIMINAL DAMAGE": 2,
           "NARCOTICS": 3,
           "ASSAULT": 4,
           "BURGLARY": 5,
           "MOTOR VEHICLE THEFT": 6,
           "ROBBERY": 7}  # can also change the values to group certain crimes into a class


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
        try:
            r = c2i[crime_type]
        except KeyError:
            r = 8  # all other types are translated to 8

        return r


    crimes["c"] = crimes["Primary Type"].apply(type2index)

    # FILTER BY CRIME TYPE
    crimes = crimes[crimes.c < 8]

    # ONE HOT ENCODING FOR THE CRIME TYPES
    ohe = np.zeros((len(crimes), 8), dtype=int)
    for i, c in enumerate(crimes.c):
        ohe[i, c] = 1

    for i, k in enumerate(c2i):
        crimes[k] = ohe[:, i]

    crimes["TOTAL"] = np.ones(len(crimes), dtype=int)

    # INCLUDE ARRESTS
    crimes["Arrest"] = crimes["Arrest"] * 1

    A = crimes[["t", "x", "y", "TOTAL", "THEFT", "BATTERY", "CRIMINAL DAMAGE", "NARCOTICS", "ASSAULT", "BURGLARY",
                "MOTOR VEHICLE THEFT", "ROBBERY", "Arrest"]].values
    # A = crimes[["t","b","TOTAL","THEFT", "BATTERY", "NARCOTICS","Arrest"]].values # is used when x and y are flattened

    B = np.zeros((t_size, A.shape[-1] - 3, y_size, x_size))

    for a in A:
        B[a[0], :, y_size - 1 - a[2], a[1]] += a[3:]  # todo normalize in channels

    #########################################################################
    #                            SAVE DATA                                  #
    #########################################################################
    save_folder = f"./data/processed/T{dT}-X{int(info['x in metres'])}M-Y{int(info['y in metres'])}M/"

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder + "plots", exist_ok=True)

    # save figures
    plt.figure(figsize=(8, 11))
    plt.scatter(X, Y, marker="x", label="range")
    plt.scatter(valid_points[:, 0], valid_points[:, 1], marker="+", label="valid")
    plt.scatter(crimes.X, crimes.Y, marker="+", label="round")
    plt.legend(loc=3, prop={"size": 20})
    plt.savefig(save_folder + "plots/" + "scatter_map.png")

    plt.figure()
    plt.title("Crimes")
    plt.imshow(grids.max(0), cmap="viridis")
    plt.savefig(save_folder + "plots/" + "crimes_max.png")
    plt.figure()
    plt.title("Demographics")
    plt.imshow(demog_grid.max(0), cmap="viridis")
    plt.savefig(save_folder + "plots/" + "demographics_max.png")
    plt.figure()
    plt.title("Street View Info")
    plt.imshow(street_grid.max(0), cmap="viridis")
    plt.savefig(save_folder + "plots/" + "street_grid_max.png")

    # save generated data
    np.savez_compressed(save_folder + "generated_data.npz",
                        crime_types_grids=B,
                        crime_grids=grids,
                        demog_grid=demog_grid,
                        street_grid=street_grid,
                        time_vectors=encode_time_vectors(t_range),
                        x_range=x_range,
                        y_range=y_range)
    pd.to_pickle(t_range, save_folder + "t_range.pkl")
    # way weather dates are too short time_vectors should be more years
    # - open weather data has more data but has quite a few gaps
    # np.save(folder+"weather_vectors.npy", weather_vectors)

    write_json(info, save_folder + "info.json")

    print("\nDONE!")
