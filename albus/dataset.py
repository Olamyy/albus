import numpy
import sys
import pandas
from pandas import compat


class Dataset:
    def __init__(self, data, **kwargs):
        """
        Dataset input handling class.

        :param data:
        :param kwargs:
        """
        if isinstance(data, dict):
            self.data = self.from_dict(data, **kwargs)
        elif isinstance(data, numpy.ndarray):
            self.data = self.from_np_array(data, **kwargs)
        elif isinstance(data, pandas.Series):
            self.data = self.from_series(data)
        else:
            self.data = data

    def from_dict(self, dict_, **kwargs):
        return pandas.DataFrame.from_dict(dict_, **kwargs)

    def from_np_array(self, array, **kwargs):
        return pandas.DataFrame(data=array, index=kwargs.get('series'), columns=kwargs.get('columns'))

    def from_series(self, series, **kwargs):
        return pandas.DataFrame(data=series, index=kwargs.get('series'), columns=kwargs.get('columns'))

    def get_null_info(self, show_all=False):
        from pandas.io.formats.format import _put_lines

        lines = []
        lines.append(self.data.index.summary())

        if len(self.data.columns) == 0:
            lines.append('Empty %s' % type(self).__name__)
            _put_lines(sys.stdout, lines)
        counts = self.data.get_dtype_counts()
        dtypes = ['%s(%d)' % k for k in sorted(compat.iteritems(counts))]
        lines.append('dtypes: %s' % ', '.join(dtypes))
        _put_lines(sys.stdout, lines)
