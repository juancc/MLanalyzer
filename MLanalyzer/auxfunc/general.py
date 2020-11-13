def batch(iterable, n=100):
    """split in batches"""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]