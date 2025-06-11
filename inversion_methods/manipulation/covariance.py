import numpy as np


def spatial_covariance(cov_space : np.ndarray,
                       sigma_space : float,
                       spatial_decay : float,
                       nbasis : int,
                       fp_data : dict,
                       ):
    
    def haversine(
            centre1, 
            centre2
            ):
        """
        This function finds the shortest path between 2 points on the surface of a sphere.
        Using the Haversine formula, this method finds the great-circle distance between
        2 lat/lon points.

        d = 2 * R * arcsin(((sin(dlat/2)**2 + cos(lat1) * cos(lat2) * (sin(dlon/2)**2))**0.5))
        """

        R = 6371 # ~radius of the Earth in km

        lat1, lon1, lat2, lon2 = map(np.deg2rad, [centre1[0], centre1[1], centre2[0], centre2[1]])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        distance = 2 * R * np.arcsin(np.sqrt(a))

        return distance
    
    bfs = fp_data[".basis"]
    bf_values = bfs.data
    bf_unique = np.unique(bf_values)
    lats = bfs.coords["lat"].data
    lons = bfs.coords["lon"].data

    bf_centres = {}
    bf_weight_centres = {}

    for bi in np.arange(nbasis):

        bf_where = np.where(bf_values==bf_unique[bi])
        bf_lat = lats[bf_where[0]]
        bf_lon = lons[bf_where[1]]

        bf_lat_weight = np.cos(np.deg2rad(bf_lat))

        bf_centres[bi] = [np.mean(bf_lat), np.mean(bf_lon)]
        bf_weight_centres[bi] = [np.average(bf_lat, weights=bf_lat_weight), np.mean(bf_lon)]
        
    for i in np.arange(nbasis - 1) + 1:
        for j in np.arange(i):
            distance = haversine(bf_weight_centres[i], bf_weight_centres[j])
            rho_space = np.exp(-distance/spatial_decay) 
            cov_space[i,j] = cov_space[j,i] = rho_space * sigma_space**2

    return cov_space