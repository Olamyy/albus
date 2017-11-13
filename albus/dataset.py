import numpy
import sys
import pandas
from pandas import compat
from pandas.core.frame import _put_str
from pandas.io.formats.printing import pprint_thing


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
        """

        :param array:
        :param kwargs:
        :return:
        """
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

        tmpl = "%s%s"

        def _verbose_repr():
            lines.append('Data columns  null_count predicted_object_type')
            space = max([len(pprint_thing(k)) for k in self.data.columns]) + 4
            counts = None

            dtypes = self.data.dtypes
            for i, col in enumerate(self.data.columns):
                dtype = dtypes.iloc[i]
                col = pprint_thing(col)
                null_col = self.data[i].isnull().sum()
                predicted_object_type = self.predict_type(numpy.random.choice(self.data[i]))
                count = ""
                lines.append(_put_str(col, space) + tmpl % (count, dtype) + "\t{}".format(null_col) +
                             "\t{}".format(predicted_object_type))

        _verbose_repr()

        _put_lines(sys.stdout, lines)

    @staticmethod
    def predict_type(value):
        if isinstance(value, numpy.int64):
            return "age"
        if isinstance(value, str):
            if value.count(',') > 2:
                return "address"
            return "name"

