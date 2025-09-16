import pandas as pd
import geopandas as gpd
import mpigeopandas

from mpi4py import MPI

world = MPI.COMM_WORLD

ds = pd.DataFrame(gpd.read_file("../data/cb_2018_us_county_500k.shp"))

ds.mpi.parallelize(world=world, scheme='single')

print(ds.attrs['mpi_indices'])