import pytest
from nemantix.core import Toolset

# import is needed because otherwise the toolset won't be registered
# noinspection PyUnusedImports
from nemantix.stl.http_requests.base import RequestsToolset

# Define a standard timeout for all toolset instances
TIMEOUT = 5


@pytest.mark.external
class TestHttpRequestsToolkit:
    # --- GET and Authentication Tests ---

    def test_http_get_basic_auth(self):
        """Verify real GET request with Basic Authentication using httpbin."""
        ts_get = Toolset.get_tool(
            tool_name="RequestsToolset.http_get", instance_args=(TIMEOUT,)
        )

        # httpbin will check that credentials match those in the URL
        url = "https://httpbin.org/basic-auth/admin/password123"
        auth_config = {"type": "basic", "username": "admin", "password": "password123"}

        result = ts_get(url=url, auth=auth_config)

        # If authentication succeeds, httpbin returns 200 and "authenticated": true
        assert "Status: 200" in result
        assert '"authenticated": true' in result
        assert '"user": "admin"' in result

    def test_http_get_bearer_token(self):
        """Verify real GET request with Bearer Token (JWT)."""
        ts_get = Toolset.get_tool(
            tool_name="RequestsToolset.http_get", instance_args=(TIMEOUT,)
        )

        url = "https://httpbin.org/bearer"
        token = "test-token-123"
        auth_config = {"type": "bearer", "token": token}

        result = ts_get(url=url, auth=auth_config)

        # Verify that httpbin correctly received and validated the token
        assert "Status: 200" in result
        assert '"authenticated": true' in result
        assert f'"{token}"' in result

    def test_http_custom_header(self):
        """Verify Custom API Key header injection in a real request."""
        ts_get = Toolset.get_tool(
            tool_name="RequestsToolset.http_get", instance_args=(TIMEOUT,)
        )

        url = "https://httpbin.org/headers"
        auth_config = {
            "type": "custom",
            "key": "X-API-KEY",
            "value": "super-secret-key",
        }

        result = ts_get(url=url, auth=auth_config)

        # httpbin returns all headers it received
        assert "Status: 200" in result
        assert '"X-Api-Key": "super-secret-key"' in result

    # --- POST Tests ---

    def test_http_post_data(self):
        """Verify POST request sends JSON data correctly over the network."""
        ts_post = Toolset.get_tool(
            tool_name="RequestsToolset.http_post", instance_args=(TIMEOUT,)
        )

        url = "https://httpbin.org/post"
        payload = {"name": "Alice", "role": "admin"}

        result = ts_post(url=url, data=payload)

        # httpbin sends back the JSON we sent within the "json" key
        assert "Status: 200" in result
        assert '"name": "Alice"' in result
        assert '"role": "admin"' in result

    # --- Error Handling Tests ---

    def test_http_error_handling(self):
        """Verify toolkit handles real DNS/connection errors gracefully."""
        ts_get = Toolset.get_tool(
            tool_name="RequestsToolset.http_get", instance_args=(TIMEOUT,)
        )

        # A non-existent domain to force a real connection failure
        url = "https://this-domain-is-completely-invalid-and-should-fail.com"

        result = ts_get(url=url)

        # Verify that the toolset catches the internal requests exception
        assert "Error" in result
        assert (
            "Failed to establish a new connection" in result
            or "Name or service not known" in result
            or "Failed to resolve" in result
        )
