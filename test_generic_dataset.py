import unittest
from datasets.flat_dataset import FlatDataGroup


class TestCrimeDateGroupInit(unittest.TestCase):
    def test_list_int(self):
        """
        Test that it can sum a list of integers
        """
        data_dim_str = "T24H-X850M-Y880M"  # needs to exist

        data_path = f"./data/processed/{data_dim_str}/"


        datagroup = FlatDataGroup(data_path=data_path)


        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)


if __name__ == '__main__':
    unittest.main()
