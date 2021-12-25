import unittest
import pandas as pd
from constants.date_time import DatetimeFreq


class TestDatetimeFreqConversions(unittest.TestCase):

    def test_hour_conversion(self):
        for freq in ['1H', 'H']:
            date_range = pd.date_range('2012-01-01', '2013-01-01', freq=freq)
            freqstr = DatetimeFreq.convert(date_range=date_range)
            self.assertEqual(freqstr, DatetimeFreq.Hour)

    def test_day_conversion(self):
        for freq in ['24H', '1D', 'D']:
            date_range = pd.date_range('2012-01-01', '2013-01-01', freq=freq)
            freqstr = DatetimeFreq.convert(date_range=date_range)
            self.assertEqual(freqstr, DatetimeFreq.Day)

    def test_week_conversion(self):
        for freq in ['1W', '168H', '7D', 'W']:
            date_range = pd.date_range('2012-01-01', '2013-01-01', freq=freq)
            freqstr = DatetimeFreq.convert(date_range=date_range)
            self.assertEqual(freqstr, DatetimeFreq.Week)

    def test_conversion_exception(self):
        date_range = pd.date_range('2012-01-01', '2013-01-01', freq='2H')
        self.assertRaises(Exception, DatetimeFreq.convert, date_range)


if __name__ == "__main__":
    unittest.main()
