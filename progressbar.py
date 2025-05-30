# progressbar.py
import sys

def progressbar(iterable, prefix='', size=60, file=sys.stdout):
    """
    Wrap an iterable with a progress bar display.

    Parameters:
        iterable (iterable): The iterable to loop over. Must have a __len__.
        prefix (str): Optional text to display before the progress bar.
        size (int): Width (in characters) of the progress bar.
        file: The output stream; defaults to sys.stdout.

    Yields:
        Each item from the input iterable.

    Example:
        >>> for item in progressbar(range(100), prefix='Processing: '):
        ...     do_something(item)
    """
    total = len(iterable)
    
    def show_progress(count):
        # Determine the number of characters to show
        filled = int(size * count / total)
        progress_str = f"{prefix}[{'#' * filled}{'.' * (size - filled)}] {count}/{total}"
        file.write(progress_str + "\r")
        file.flush()
    
    show_progress(0)
    for count, item in enumerate(iterable, start=1):
        yield item
        show_progress(count)
    file.write("\n")
    file.flush()

# Optional: If you run this module directly, show an example progress bar.
if __name__ == '__main__':
    import time
    for i in progressbar(range(100), prefix='Loading: '):
        time.sleep(0.05)
    print("Done!")
