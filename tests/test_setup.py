def test_core_dependencies_are_available():
    """Verify that the project's main dependencies can be imported."""

    import numpy
    import pandas
    import scipy
    import sklearn
    import streamlit

    assert numpy.__version__
    assert pandas.__version__
    assert scipy.__version__
    assert sklearn.__version__
    assert streamlit.__version__