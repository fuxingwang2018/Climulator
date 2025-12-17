from datetime import datetime, timedelta

def generate_time_series(start_year=2000, end_year=2009, step_hours=6):
    """
    Generate a list of datetime objects from start_year-01-01 00:00
    to end_year-12-31 18:00 with a given step in hours.
    """
    start_time = datetime(start_year, 1, 1, 0, 0)
    end_time = datetime(end_year + 1, 1, 1, 0, 0)  # up to next year
    delta = timedelta(hours=step_hours)
    
    times = []
    t = start_time
    while t < end_time:
        times.append(t)
        t += delta
    return times


def get_time_indices(times, start_date_str, end_date_str, date_format="%Y-%m-%d %H:%M"):
    """
    Get start and end indices from a list of datetime objects.

    Parameters
    ----------
    times : list of datetime
        The full time list.
    start_date_str : str
        Start date in the format of `date_format`, e.g. '2005-07-01 00:00'.
    end_date_str : str
        End date in the format of `date_format`, e.g. '2005-08-31 24:00'.
    date_format : str
        Format used to parse input date strings.

    Returns
    -------
    start_idx, end_idx : int
        The indices of the start and end time records (inclusive).
    """
    # Handle 24:00 edge case (convert to next day 00:00)
    def parse_date(s):
        if "24:00" in s:
            d = datetime.strptime(s.replace(" 24:00", " 00:00"), date_format)
            d += timedelta(days=1)
            return d
        return datetime.strptime(s, date_format)
    
    start_dt = parse_date(start_date_str)
    end_dt = parse_date(end_date_str)

    # Find indices
    start_idx = next(i for i, t in enumerate(times) if t >= start_dt)
    end_idx = next(i for i, t in enumerate(times) if t >= end_dt) - 1  # inclusive

    return start_idx, end_idx



