def chunks(filename, buffer_size=4096):
    """Reads `filename` in chunks of `buffer_size` bytes and yields each chunk
    until no more characters can be read; the last chunk will most likely have
    less than `buffer_size` bytes.

    :param str filename: Path to the file
    :param int buffer_size: Buffer size, in bytes (default is 4096)
    :return: Yields chunks of `buffer_size` size until exhausting the file
    :rtype: str

    """
    with open(filename, "r") as fp:
        chunk = fp.read(buffer_size)
        chunk = '{\"data\":[' + chunk
        while chunk:
            chunk = chunk.replace('}{\"query\":', '},{\"query\":')
            if("Reeceistheword" in chunk):
                print(chunk)
            yield chunk
            chunk = fp.read(buffer_size)

def chars(filename, buffersize=4096):
    """Yields the contents of file `filename` character-by-character. Warning:
    will only work for encodings where one character is encoded as one byte.

    :param str filename: Path to the file
    :param int buffer_size: Buffer size for the underlying chunks,
    in bytes (default is 4096)
    :return: Yields the contents of `filename` character-by-character.
    :rtype: char

    """
    for chunk in chunks(filename, buffersize):
        for char in chunk:
            yield char

def main(buffersize, filenames):
    """Reads several files character by character and redirects their contents
    to `/dev/null`.

    """
    for filename in filenames:
        with open(filename+"_json", "w") as fp:
            for char in chars(filename, buffersize):
                fp.write(char)

        with open(filename+"_json", "a") as fp:
                fp.write(']}')

if __name__ == "__main__":
    # Try reading several files varying the buffer size
    import sys
    buffersize = int(sys.argv[1])
    filenames  = sys.argv[2:]
    sys.exit(main(buffersize, filenames))
