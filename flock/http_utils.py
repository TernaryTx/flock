from __future__ import annotations

import requests

# Response header naming the UniProtKB release a request was served from, e.g.
# '2026_02'. The release moves every few weeks and every entry can move with it,
# so it is the honest provenance key - a download date only says when we asked.
UNIPROT_RELEASE_HEADER = 'X-UniProt-Release'


def stream_to_file(url: str, path: str, timeout: int = 60, chunk_size: int = 1 << 20) -> str:
    """Streams a URL to a local file, stripping only its HTTP transfer encoding.

    Args:
        url: Source URL.
        path: Destination path.
        timeout: Seconds to wait for the connection and for each read. This
            bounds a stalled transfer, not a slow one - the reviewed table takes
            around two minutes to arrive and must not be cut off.
        chunk_size: Bytes to write at a time.

    Returns:
        The UniProt release the response was served from, empty if the server
        did not name one.

    Raises:
        requests.HTTPError: If the server returns an error status.
    """
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(path, 'wb') as handle:
            # The response arrives double-gzipped: UniProt gzips the body for
            # compressed=true, then gzips it again as Content-Encoding because
            # requests advertises Accept-Encoding: gzip. iter_content strips the
            # transport layer, leaving the single gzip read_local_data expects.
            # Writing the raw bytes instead would store an unreadable nesting.
            for chunk in response.iter_content(chunk_size=chunk_size):
                handle.write(chunk)
        return response.headers.get(UNIPROT_RELEASE_HEADER, '')
