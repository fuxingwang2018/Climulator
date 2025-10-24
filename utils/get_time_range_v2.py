from datetime import datetime, timedelta

def generate_time_series(start_date, end_date, step_hours=6, date_format="%Y-%m-%d %H:%M"):
    """
    Generate a list of datetime objects from start_date to end_date (inclusive)
    with a given step in hours.

    Parameters
    ----------
    start_date : str or datetime
        Start date (e.g., '2000-01-01 00:00').
    end_date : str or datetime
        End date (e.g., '2009-12-31 18:00').
    step_hours : int
        Time step in hours (default: 6).
    date_format : str
        Format for parsing date strings if input is str.

    Returns
    -------
    times : list of datetime
        List of datetime objects spaced by step_hours.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, date_format)
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, date_format)

    delta = timedelta(hours=step_hours)
    times = []
    t = start_date
    while t <= end_date:
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


