from datetime import datetime, timedelta

def parse_datetime_safe(date_input, date_format="%Y-%m-%d %H:%M"):
    """
    Safely parse a datetime string or object, supporting '24:00' as next-day 00:00.
    """
    if isinstance(date_input, datetime):
        return date_input
    if "24:00" in date_input:
        # Convert "YYYY-MM-DD 24:00" to next day "YYYY-MM-(DD+1) 00:00"
        temp = datetime.strptime(date_input.replace(" 24:00", " 00:00"), date_format)
        return temp + timedelta(days=1)
    return datetime.strptime(date_input, date_format)


def generate_time_series(start_date, end_date, step_hours=6, date_format="%Y-%m-%d %H:%M"):
    """
    Generate a list of datetime objects from start_date to end_date (inclusive)
    with a given step in hours.
    """
    start_date = parse_datetime_safe(start_date, date_format)
    end_date = parse_datetime_safe(end_date, date_format)

    delta = timedelta(hours=step_hours)
    times = []
    t = start_date
    while t <= end_date:
        times.append(t)
        t += delta
    return times


def get_time_indices(times, start_date, end_date, date_format="%Y-%m-%d %H:%M"):
    """
    Get start and end indices from a list of datetime objects.

    Parameters
    ----------
    times : list of datetime
        The full time list.
    start_date, end_date : str or datetime
        Start and end date defining the selection period.
    date_format : str
        Format used to parse input date strings.

    Returns
    -------
    start_idx, end_idx : int
        The indices of the start and end time records (inclusive).
    """
    start_dt = parse_datetime_safe(start_date, date_format)
    end_dt = parse_datetime_safe(end_date, date_format)

    # Find indices (inclusive range)
    start_idx = next(i for i, t in enumerate(times) if t >= start_dt)
    end_idx = next(i for i, t in enumerate(times) if t > end_dt) - 1

    return start_idx, end_idx

