import six
import h5py
import numpy as np
import scipy.sparse as ss


FORMAT_DICT = {
    'csr': ss.csr_matrix,
    'csc': ss.csc_matrix,
    'coo': ss.coo_matrix,
}

indptr_dtype = np.int64
indices_dtype = np.int64
row_dtype = np.int64
col_dtype = np.int64


def get_format_str(data):
    for format_str, format_class in six.viewitems(FORMAT_DICT):
        if isinstance(data, format_class):
            return format_str
    raise ValueError("Data type {} is not supported.".format(type(data)))


def get_format_class(format_str):
    try:
        format_class = FORMAT_DICT[format_str]
    except KeyError:
        raise ValueError("Format {} is not supported."
                         .format(format_str))
    return format_class


def is_compressed_format(format_str):
    return format_str in ('csc', 'csr')


class Group(h5py.Group):
    """The HDF5 group that can detect and create sparse matrix.
    """

    def __getitem__(self, key):
        h5py_item = super(Group, self).__getitem__(key)
        if isinstance(h5py_item, h5py.Group):
            if 'h5sparse_format' in h5py_item.attrs:
                # detect the sparse matrix
                return Dataset(h5py_item)
            else:
                return Group(h5py_item.id)
        elif isinstance(h5py_item, h5py.Dataset):
            return h5py_item
        else:
            raise ValueError("Unexpected item type.")

    def create_dataset_compressed(self, name, sparse_format, shape, data, indices, indptr,
                                  dtype, **kwargs):
        """Create a dataset in csc or csr format"""
        assert sparse_format in ("csc", "csr")

        group = self.create_group(name)
        group.attrs['h5sparse_format'] = sparse_format
        group.attrs['h5sparse_shape'] = shape
        group.create_dataset('data', data=data, dtype=dtype, **kwargs)
        group.create_dataset('indices', data=indices, dtype=indices_dtype, **kwargs)
        group.create_dataset('indptr', data=indptr, dtype=indptr_dtype, **kwargs)
        return group

    def create_dataset_coo(self, name, sparse_format, shape, data, row, col,
                           dtype, **kwargs):
        """Create a dataset in csc or csr format"""
        assert sparse_format == "coo"

        group = self.create_group(name)
        group.attrs['h5sparse_format'] = sparse_format
        group.attrs['h5sparse_shape'] = shape
        group.create_dataset('data', data=data, dtype=dtype, **kwargs)
        group.create_dataset('row', data=row, dtype=row_dtype, **kwargs)
        group.create_dataset('col', data=col, dtype=col_dtype, **kwargs)
        return group

    def create_dataset_from_dataset(self, name, data, dtype, **kwargs):
        sparse_format = data.attrs['h5sparse_format']
        if (is_compressed_format(sparse_format)):
            group = self.create_dataset_compressed(name,
                                                   data.attrs['h5sparse_format'],
                                                   data.attrs['h5sparse_shape'],
                                                   data.h5py_group['data'],
                                                   data.h5py_group['indices'],
                                                   data.h5py_group['indptr'],
                                                   dtype,
                                                   **kwargs)
        else:
            group = self.create_dataset_coo(name,
                                            data.attrs['h5sparse_format'],
                                            data.attrs['h5sparse_shape'],
                                            data.h5py_group['data'],
                                            data.h5py_group['row'],
                                            data.h5py_group['col'],
                                            dtype,
                                            **kwargs)
        return group

    def create_dataset_from_scipy(self, name, data, dtype, **kwargs):
        sparse_format = get_format_str(data)
        if (is_compressed_format(sparse_format)):
            group = self.create_dataset_compressed(name,
                                                   sparse_format,
                                                   data.shape,
                                                   data.data,
                                                   data.indices,
                                                   data.indptr,
                                                   dtype,
                                                   **kwargs)
        else:
            group = self.create_dataset_coo(name,
                                            sparse_format,
                                            data.shape,
                                            data.data,
                                            data.row,
                                            data.col,
                                            dtype,
                                            **kwargs)
        return group

    def create_dataset(self, name, shape=None, dtype=None, data=None,
                       sparse_format=None, **kwargs):
        """Create 3 datasets in a group to represent the sparse array.

        Parameters
        ----------
        sparse_format:
        """
        if isinstance(data, Dataset):
            assert sparse_format is None
            group = self.create_dataset_from_dataset(name, data, dtype, **kwargs)
        elif ss.issparse(data):
            if sparse_format is not None:
                format_class = get_format_class(sparse_format)
                data = format_class(data)
            group = self.create_dataset_from_scipy(name,
                                                   data,
                                                   dtype,
                                                   **kwargs)
        elif data is None and sparse_format is not None:
            format_class = get_format_class(sparse_format)
            if dtype is None:
                dtype = np.float64
            if shape is None:
                shape = (0, 0)
            data = format_class(shape, dtype=dtype)
            group = self.create_dataset_from_scipy(name,
                                                   data,
                                                   dtype,
                                                   **kwargs)
        else:
            # forward the arguments to h5py
            assert sparse_format is None
            return super(Group, self).create_dataset(
                name, data=data, shape=shape, dtype=dtype, **kwargs)
        return Dataset(group)


class File(h5py.File, Group):
    """The HDF5 file object that can detect and create sparse matrix.
    """
    pass


class Dataset(h5py.Group):
    """The HDF5 sparse matrix dataset.

    Parameters
    ----------
    h5py_group : h5py.Dataset
    """

    def __init__(self, h5py_group):
        super(Dataset, self).__init__(h5py_group.id)
        self.h5py_group = h5py_group
        self.shape = tuple(self.attrs['h5sparse_shape'])
        self.format_str = self.attrs['h5sparse_format']
        self.dtype = h5py_group['data'].dtype

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.step is not None:
                raise NotImplementedError("Index step is not supported.")
            start = key.start
            stop = key.stop
            if stop is not None and stop > 0:
                stop += 1
            if start is not None and start < 0:
                start -= 1
            indptr_slice = slice(start, stop)
            indptr = self.h5py_group['indptr'][indptr_slice]
            data = self.h5py_group['data'][indptr[0]:indptr[-1]]
            indices = self.h5py_group['indices'][indptr[0]:indptr[-1]]
            indptr -= indptr[0]
            if self.format_str == 'csr':
                shape = (indptr.size - 1, self.shape[1])
            elif self.format_str == 'csc':
                shape = (self.shape[0], indptr.size - 1)
            else:
                raise NotImplementedError("Slicing for format {} is not implemented."
                                          .format(self.format_str))
            format_class = get_format_class(self.attrs['h5sparse_format'])
            return format_class((data, indices, indptr), shape=shape)
        elif isinstance(key, tuple) and key == ():
            if (is_compressed_format(self.format_str)):
                data = self.h5py_group['data'][()]
                indices = self.h5py_group['indices'][()]
                indptr = self.h5py_group['indptr'][()]
                shape = self.shape
                format_class = get_format_class(self.attrs['h5sparse_format'])
                return format_class((data, indices, indptr), shape=shape)
            else:
                data = self.h5py_group['data'][()]
                row = self.h5py_group['row'][()]
                col = self.h5py_group['col'][()]
                shape = self.shape
                format_class = get_format_class(self.attrs['h5sparse_format'])
                return format_class((data, (row, col)), shape=shape)
        else:
            raise NotImplementedError("Only support one slice as index.")

    @property
    def value(self):
        return self[()]

    def append(self, sparse_matrix):
        if self.format_str != get_format_str(sparse_matrix):
            raise ValueError("Format not the same.")

        if self.format_str == 'csr':
            # data
            data = self.h5py_group['data']
            orig_data_size = data.shape[0]
            new_shape = (orig_data_size + sparse_matrix.data.shape[0],)
            data.resize(new_shape)
            data[orig_data_size:] = sparse_matrix.data

            # indptr
            indptr = self.h5py_group['indptr']
            orig_data_size = indptr.shape[0]
            append_offset = indptr[-1]
            new_shape = (orig_data_size + sparse_matrix.indptr.shape[0] - 1,)
            indptr.resize(new_shape)
            indptr[orig_data_size:] = (sparse_matrix.indptr[1:].astype(np.int64)
                                       + append_offset)

            # indices
            indices = self.h5py_group['indices']
            orig_data_size = indices.shape[0]
            new_shape = (orig_data_size + sparse_matrix.indices.shape[0],)
            indices.resize(new_shape)
            indices[orig_data_size:] = sparse_matrix.indices

            # shape
            self.shape = (
                self.shape[0] + sparse_matrix.shape[0],
                max(self.shape[1], sparse_matrix.shape[1]))
            self.attrs['h5sparse_shape'] = self.shape
        else:
            raise NotImplementedError("The append method for format {} is not implemented."
                                      .format(self.format_str))
