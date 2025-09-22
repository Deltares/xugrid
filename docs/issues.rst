Known Issues
============

This page documents issues known to exist in ``xugrid``. Many are related
to the way ``xugrid`` extends ``xarray``, and may be addressed by `introducing
a custom index <https://github.com/Deltares/xugrid/issues/35>`_. For instance:

1. `Arithmetic is not commutative <https://github.com/Deltares/xugrid/issues/34>`_.
2. `Operations (e.g. groupby) <https://github.com/Deltares/xugrid/issues/181>`_ do not always return ``xugrid`` types.
3. Handling of NetCDF default fill values `can be tricky <https://github.com/pydata/xarray/issues/2742>`_.