
import pandas as pd
import geopandas as gpd
import mpigeopandas

from mpi4py import MPI

world = MPI.COMM_WORLD

ds = pd.DataFrame(gpd.read_file("../data/cb_2018_us_county_500k.shp"))

print(ds['geometry'])

ds.mpi.parallelize(world=world, scheme='chunked', increment=10)

# df.mpi.parallel_scheme = 'single'

# ds.mpi.to_netcdf("test.nc", mode='w', format='NETCDF4', engine='h5netcdf')


