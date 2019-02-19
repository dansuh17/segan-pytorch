import unittest
import numpy as np
import emph


class TestEmphasis(unittest.TestCase):
    def test_pre_emphasis(self):
        """
        Tests equality after de-emphasizing pre-emphasized signal.
        """
        rand_signal_batch = np.random.randint(low=1, high=10, size=(10, 1, 400))
        reconst_batch = emph.de_emphasis(emph.pre_emphasis(rand_signal_batch))

        # after de-emphasis, the signal must have been restored
        self.assertEqual(rand_signal_batch.shape, reconst_batch.shape)
        self.assertTrue(np.allclose(rand_signal_batch, reconst_batch))


if __name__ == '__main__':
    unittest.main()
