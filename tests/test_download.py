from unittest.mock import Mock
import io
import pytest
from ct_poc.acquire import range_read

def response(status, content, content_range):
    result = Mock()
    result.status_code = status
    result.headers = {"Content-Range": content_range}
    result.raw = io.BytesIO(content)
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result

def test_refuses_accidental_full_archive():
    session = Mock()
    session.get.return_value = response(200, b"large archive", "")
    with pytest.raises(RuntimeError, match="full archive"):
        range_read(session, 0, 512, "etag")

def test_rejects_wrong_range():
    session = Mock()
    session.get.return_value = response(206, b"abc", "bytes 1-3/100")
    with pytest.raises(RuntimeError, match="Content-Range"):
        range_read(session, 0, 3, "etag")

def test_rejects_truncation():
    session = Mock()
    session.get.return_value = response(206, b"ab", "bytes 0-2/100")
    with pytest.raises(RuntimeError, match="Incomplete"):
        range_read(session, 0, 3, "etag")
