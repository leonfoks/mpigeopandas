import numpy as np
from pandas.api.extensions import register_dataframe_accessor
import mpi_utilities as mpiu
from mpi4py import MPI

@register_dataframe_accessor('mpi')
class mpi_DataFrame:
    def __init__(self, obj):
        self._obj = obj

    @property
    def world(self):
        return self._obj.__world

    @world.setter
    def world(self, value):
        assert isinstance(value, MPI.Comm), ValueError("world must be an MPI communicator")
        self._obj.__world = value

    @property
    def is_mpi(self):
        return self.world is not None

    @property
    def mpi_scheme(self):
        return self._obj.__mpi_scheme

    @mpi_scheme.setter
    def mpi_scheme(self, value):
        schemes = ('queue', 'single', 'chunked')
        assert value in schemes, ValueError("parallel scheme must be in {schemes}")
        self.__mpi_scheme = value

    @property
    def mpi_indices(self):
        return self._obj.attrs['mpi_indices']

    @mpi_indices.setter
    def mpi_indices(self, value):
        assert isinstance(value, np.ndarray), TypeError("mpi_indices must be a numpy array")
        self._obj.attrs['mpi_indices'] = value

    def parallelize(self, world, scheme, increment=None):
        self.world = world

        self.mpi_scheme = scheme

        n_rows = self._obj.shape[0]
        match scheme:
            case 'queue':
                self.mpi_indices = np.arange(n_rows)
            case 'single':
                s, c = mpiu.load_balance(n_rows, world.size)
                self.mpi_indices = np.stack([s, s+c], axis=1)
            case 'chunked':
                mpi_indices = np.arange(0, n_rows, increment)
                s, c = mpiu.load_balance(n_rows, world.size)
                print(mpi_indices)
                print(s, c)


        return self._obj
