import numpy as np
import mlstack.metrics as M


def test_ccc():
    x = np.random.rand(20)
    y = np.random.rand(20)

    c1 = M.ccc(x, y)
    c2 = M.ccc2(x, y)

    print(c1)
    print(c2)

    assert np.isclose(c1, c2)
