"""Simple logging module"""

import io
import sys
from datetime import datetime


class Log():
    stdout = sys.stdout
    stderr = sys.stderr
    session = 'default'
    cache = {
        session: io.StringIO()
    }


    @staticmethod
    def set(session):
        """Initialize new logging session.

        Parameters
        ----------
        session : str or None
            Session name.
        
        Returns
        -------
        None

        """
        Log.session = session
        if not Log.session in Log.cache:
            Log.cache[Log.session] = io.StringIO()
        sys.stdout = Log.cache[Log.session]
        sys.stderr = Log.cache[Log.session]


    @staticmethod
    def reset():
        sys.stdout = Log.stdout
        sys.stderr = Log.stderr


    @staticmethod
    def get(session=None):
        """Get list of log messages for given session name od 'default' if name is not provided.

        Parameters
        ----------
        session : str
            Session name.
        
        Returns
        -------
        list[str]
            List of session messages.

        """
        if session in Log.cache:
            return Log.cache[session].getvalue()
        else:
            return Log.cache[Log.session].getvalue()


    @staticmethod
    def clear(session=None):
        """Clear session with given name or 'default' if name is not provided.

        Parameters
        ----------
        session : str
            Session name.
        
        Returns
        -------
        None

        """
        if session in Log.cache:
            Log.cache[session].flush()
        else:
            Log.cache[Log.session].flush()


    @staticmethod
    def clear_all():
        """Clear session cache.
        
        Returns
        -------
        None

        """
        Log.cache = {}
        set('default')


    @staticmethod
    def save(file_path, session=None):
        """Save session to given file.

        Parameters
        ----------
        file_path : str
            File path.

        session : str
            Session name.

        Returns
        -------
        None

        """
        with open(file_path, 'w', encoding="utf-8") as file:
            file.write(Log.get(session))


    @staticmethod
    def print(*args, **kwargs):
        """Print message using `print` function and store it to the current session.
        
        Returns
        -------
        None

        """
        # Save `file` keyword argument.
        file = kwargs.get('file')

        # Print to session buffer.
        kwargs['file'] = Log.cache[Log.session]
        print(*args, **kwargs)

        # Restore `file` keyword argument.
        if file:
            kwargs['file'] = file
        else:
            del kwargs['file']

        # Print to stdout.
        sys.stdout = Log.stdout
        sys.stderr = Log.stderr
        print(*args, **kwargs)
        sys.stdout = Log.cache[Log.session]
        sys.stderr = Log.cache[Log.session]


    @staticmethod
    def printt(*args, **kwargs):
        """Print message with timestamp prefix using `print` function and store it to the current session.
        
        Returns
        -------
        None

        """
        Log.print(f'[{datetime.fromtimestamp(datetime.timestamp(datetime.now()))}]', *args, **kwargs)


    @staticmethod
    def info(*args, **kwargs):
        """Print INFO message with timestamp prefix using `print` function and store it to the current session.
        
        Returns
        -------
        None

        """
        Log.printt('[INFO]', *args, **kwargs)


    @staticmethod
    def debug(*args, **kwargs):
        """Print DEBUG message with timestamp prefix using `print` function and store it to the current session.
        
        Returns
        -------
        None

        """
        Log.printt('[DEBUG]', *args, **kwargs)


    @staticmethod
    def warning(*args, **kwargs):
        """Print WARNING message with timestamp prefix using `print` function and store it to the current session.
        
        Returns
        -------
        None

        """
        Log.printt('[WARNING]', *args, **kwargs)


    @staticmethod
    def error(*args, **kwargs):
        """Print ERROR message with timestamp prefix using `print` function and store it to the current session.
        
        Returns
        -------
        None

        """
        Log.printt('[ERROR]', *args, **kwargs)
