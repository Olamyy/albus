import numpy


class Imputer(object):
    def __init__(self):
        """
        Attributes
        ----------

        """

    def drop(self, x, condition):
        """ Drops all observations that have missing data

        Parameters
        ----------
        x : numpy.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features
        condition : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        """

        # drop observations with missing values
        return x[numpy.sum(condition(x), axis=1) == 0]

    def replace(self, x, condition, in_place=False):
        """ Replace missing data with a random observation with data

        Parameters
        ----------
        x : numpy.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
            :param in_place:
        """
        if in_place:
            data = x
        else:
            data = numpy.copy(x)

        for col in range(x.shape[1]):
            nan_ids = condition(x[:, col])
            val_ids = numpy.random.choice(numpy.where(~nan_ids)[0], numpy.sum(nan_ids))
            data[nan_ids, col] = data[val_ids, col]
        return data

    def summarize(self, x, summary_func, missing_data_cond, in_place=False):
        """
        Substitutes missing values with a statistical summary of each
        feature vector

        Parameters
        ----------
        x : numpy.array
            Assumes that each feature column is of single type. Converts
            digit string features to float.
        summary_func : function
            Summarization function to be used for imputation
            (mean, median, mode, max, min...)
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        """

        if in_place:
            data = x
        else:
            data = numpy.copy(x)

        # replace missing values with the summarization function
        for col in range(x.shape[1]):
            nan_ids = missing_data_cond(x[:, col])
            if True in nan_ids:
                val = summary_func(x[~nan_ids, col])
                data[nan_ids, col] = val

        return data

    def one_hot(self, x, missing_data_cond, weighted=False, in_place=False):
        """Create a one-hot row for each observation

        Parameters
        ----------
        x : numpy.ndarray
            Matrix with categorical data, where rows are observations and
            columns are features
        missing_data_cond : function
            Method that takes one value and returns True if it represents
            missing data or false otherwise.
        weighted : bool
            Replaces one-hot by n_classes-hot.

        Returns
        -------
        data : numpy.ndarray
            Matrix with categorical data replaced with one-hot rows
        """

        if in_place:
            data = x
        else:
            data = numpy.copy(x)

        # find rows and columns with missing data
        _, miss_cols = numpy.where(missing_data_cond(data))
        miss_cols_uniq = numpy.unique(miss_cols)

        for miss_col in miss_cols_uniq:
            uniq_vals, indices = numpy.unique(data[:, miss_col],
                                              return_inverse=True)
            if weighted:
                data = numpy.column_stack((data, numpy.eye(uniq_vals.shape[0],
                                                           dtype=int)[indices] * uniq_vals.shape[0]))
            else:
                data = numpy.column_stack((data, numpy.eye(uniq_vals.shape[0],
                                                           dtype=int)[indices]))

        # remove categorical columns with missing data
        data = numpy.delete(data, miss_cols, 1)
        return data
