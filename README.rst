
.. image:: https://github.com/deltares/xugrid/raw/main/docs/_static/xugrid.svg
  :target: https://github.com/deltares/xugrid

Xugrid
======

**This is a work in progress.** `See documentation <https://deltares.github.io/xugrid/>`_.

Xarray extension to work with 2D unstructured grids, for data and topology
stored according to `UGRID conventions
<https://ugrid-conventions.github.io/ugrid-conventions>`_.

Processing structured data with xarray is convenient and efficient. The goal
of Xugrid (pronounced "kiss you grid" by `visionaries 🗢
<https://github.com/visr>`_ ) is to extend this ease to unstructured grids.

.. code:: python

    import matplotlib.pyplot as plt
    import xugrid

    uda = xugrid.data.elevation_nl()
    sections = xugrid.data.cross_sections_nl()

    section_data = uda.along_line(sections)

    fig, (ax0, ax1) = plt.subplots(ncols=2)
    uda.plot(ax=ax0, cmap="terrain")
    sections.plot(ax=ax0)
    section_data.plot.line(ax=ax1)


Installation
------------

.. code:: console

   pip install xugrid
