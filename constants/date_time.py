from pandas.core.indexes.datetimes import DatetimeIndex

HOURS_IN_DAY = 24
HOURS_IN_WEEK = 168
HOURS_IN_YEAR = 8760


class TemporalVariables:
    Hour = "Hour"
    DayOfWeek = "Day of Week"
    TimeOfMonth = "Time of Month"
    TimeOfYear = "Time of Year"


class DatetimeFreq:
    Hour = '1H'
    Day = '24H'
    Week = '168H'

    @classmethod
    def convert(cls, date_range: DatetimeIndex):
        """
        converts datetime_ranges to DatetimeFreq - because datetime_range.freq_str is cached property and does not
        always return constant str
        :param date_range: pandas DatetimeIndex
        :return:
        """
        # datetime_range.freqstr is cached property and does not always return constant str
        freq_str = str(date_range.freq)
        if freq_str in ('<Hour>', 'H', '1H'):
            return cls.Hour
        elif freq_str in ('<Day>', '<24 * Hours>', 'D', '1D', '24H'):
            return cls.Day
        elif freq_str in ('<Week>', '<Week: weekday=6>', '<7 * Days>', '<168 * Hours>', '7D', '168H', '1W', 'W'):
            return cls.Week
        else:
            raise Exception(f"freq_str '{freq_str}' from datetime_range cannot not be converted into a valid "
                            f"DatetimeFreq enum.")
