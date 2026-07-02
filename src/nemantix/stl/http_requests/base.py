import requests
import json
from typing import Optional, Dict, Any
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
from nemantix.core import tool, Toolset


class RequestsToolset(Toolset):
    """
    A stateless HTTP toolset where authentication is passed explicitly
    in every request via the 'auth' parameter.
    """

    def __init__(self, timeout: int = 10, user_agent: str = "Agent/1.0"):
        """
        Initialize the HTTP session settings.

        Args:
            timeout (int): Global timeout for requests in seconds.
            user_agent (str): User-Agent header string.
        """
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout = timeout

    # --- Helper: Auth Logic ---

    def _apply_auth(
        self, auth_params: Optional[Dict[str, str]], headers: Dict[str, str]
    ) -> Optional[Any]:
        """
        Parses the 'auth' dictionary and returns the requests.auth object
        OR updates the headers dict in-place.
        """
        if not auth_params:
            return None

        auth_type = auth_params.get("type", "").lower()

        if auth_type == "basic":
            return HTTPBasicAuth(auth_params["username"], auth_params["password"])

        elif auth_type == "digest":
            return HTTPDigestAuth(auth_params["username"], auth_params["password"])

        elif auth_type == "bearer":
            # Updates headers directly, returns None for the 'auth' kwarg
            headers["Authorization"] = f"Bearer {auth_params['token']}"
            return None

        elif auth_type == "custom":
            # For API Keys like "X-API-KEY: 12345"
            headers[auth_params["key"]] = auth_params["value"]
            return None

        else:
            raise ValueError(f"Unsupported auth type: {auth_type}")

    # --- The Tools ---

    @tool
    def http_get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        auth: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Performs an HTTP GET request to retrieve data.

        Args:
            url (str): The URL to request.
            params (Dict[str, Any], optional): dictionary of query parameters. Defaults to None.
            auth (Dict[str, str], optional): Auth config. Examples:
                  {"type": "basic", "username": "user", "password": "pass"}
                  {"type": "bearer", "token": "jwt_token"}
                  {"type": "custom", "key": "X-API-KEY", "value": "123"}

        Returns:
            str: The response status, URL, and body content.

        Example call:
            http_get(
                url="https://api.example.com/users",
                params={"limit": 10},
                auth={"type": "bearer", "token": "abc-123"}
            )
        """
        return self._make_request("GET", url, params=params, auth_config=auth)

    @tool
    def http_post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        auth: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Performs an HTTP POST request to submit data.

        Args:
            url (str): The URL to request.
            data (Dict[str, Any], optional): The JSON body to send. Defaults to None.
            auth (Dict[str, str], optional): Auth config (see http_get for examples).

        Returns:
            str: The response status, URL, and body content.

        Example call:
            http_post(
                url="https://api.example.com/submit",
                data={"name": "Alice", "role": "admin"},
                auth={"type": "custom", "key": "X-API-Key", "value": "secret"}
            )
        """
        return self._make_request("POST", url, json_data=data, auth_config=auth)

    @tool
    def http_put(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        auth: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Performs an HTTP PUT request to update data.

        Args:
            url (str): The URL to request.
            data (Dict[str, Any], optional): The JSON body to send. Defaults to None.
            auth (Dict[str, str], optional): Auth config.

        Returns:
            str: The response status, URL, and body content.

        Example call:
            http_put(
                url="https://api.example.com/items/42",
                data={"status": "archived"},
                auth={"type": "basic", "username": "user", "password": "pw"}
            )
        """
        return self._make_request("PUT", url, json_data=data, auth_config=auth)

    @tool
    def http_delete(self, url: str, auth: Optional[Dict[str, str]] = None) -> str:
        """
        Performs an HTTP DELETE request to remove a resource.

        Args:
            url (str): The URL to request.
            auth (Dict[str, str], optional): Auth config.

        Returns:
            str: The response status and body content.

        Example call:
            http_delete(
                url="https://api.example.com/items/42",
                auth={"type": "bearer", "token": "xyz-987"}
            )
        """
        return self._make_request("DELETE", url, auth_config=auth)

    # --- Internal Request Handler ---

    def _make_request(
        self, method: str, url: str, params=None, json_data=None, auth_config=None
    ) -> str:
        # Prepare headers (copy existing session headers to avoid mutation)
        req_headers = dict(self.session.headers)

        try:
            # Calculate Auth (updates headers OR returns auth object)
            request_auth_obj = self._apply_auth(auth_config, req_headers)

            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                headers=req_headers,
                auth=request_auth_obj,
                timeout=self.timeout,
            )

            try:
                content = response.json()
                content_str = json.dumps(content, indent=2)
            except json.JSONDecodeError:
                content_str = response.text

            return (
                f"Status: {response.status_code}\n"
                f"URL: {response.url}\n"
                f"Response:\n{content_str}"
            )

        except Exception as e:
            return f"Error executing {method} request: {str(e)}"
