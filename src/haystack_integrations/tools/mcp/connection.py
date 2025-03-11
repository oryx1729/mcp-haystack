import asyncio
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from contextlib import AsyncExitStack
from dataclasses import dataclass, fields
from typing import Any, cast

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from haystack import logging
from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.tools import Tool
from haystack.tools.errors import ToolInvocationError

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base class for MCP-related errors."""

    pass


class MCPConnectionError(MCPError):
    """Error connecting to MCP server."""

    pass


class MCPToolNotFoundError(MCPError):
    """Error when a tool is not found on the server."""

    pass


class MCPResponseTypeError(MCPError):
    """Error when response content type is not supported."""

    pass


class MCPInvocationError(ToolInvocationError):
    """Error during tool invocation."""

    pass


class MCPClient(ABC):
    """
    Abstract base class for MCP clients.

    This class defines the common interface and shared functionality for all MCP clients,
    regardless of the transport mechanism used.
    """

    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.stdio: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception] | None = None
        self.write: MemoryObjectSendStream[types.JSONRPCMessage] | None = None

    @abstractmethod
    async def connect(self) -> list[Tool]:
        """
        Connect to an MCP server.

        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        pass

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """
        Call a tool on the connected MCP server.

        :param tool_name: Name of the tool to call
        :param tool_args: Arguments to pass to the tool
        :returns: Result of the tool invocation
        :raises MCPError: If not connected to an MCP server
        :raises MCPInvocationError: If the tool invocation fails
        :raises MCPResponseTypeError: If response type is not TextContent
        """
        if not self.session:
            message = "Not connected to an MCP server"
            raise MCPError(message)

        try:
            result = await self.session.call_tool(tool_name, tool_args)
            validated_result = self._validate_response(tool_name, result)
            return validated_result
        except MCPError:
            # Re-raise specific MCP errors directly
            raise
        except Exception as e:
            # Wrap other exceptions with context about which tool failed
            message = f"Failed to invoke tool '{tool_name}': {e!s}"
            raise MCPInvocationError(message) from e

    def _validate_response(self, tool_name: str, result: types.CallToolResult) -> types.CallToolResult:
        """
        Validate response from an MCP tool call, accepting only TextContent.

        :param tool_name: Name of the called tool (for error messages)
        :param result: CallToolResult from MCP tool call
        :returns: The original CallToolResult object
        :raises MCPResponseTypeError: If content type is not TextContent
        :raises MCPInvocationError: If the tool call resulted in an error
        """

        # Check for error response
        if result.isError:
            if len(result.content) > 0 and isinstance(result.content[0], types.TextContent):
                # Get the error message from the first item
                first_item = result.content[0]
                message = f"Tool '{tool_name}' returned an error: {first_item.text}"
            else:
                message = f"Tool '{tool_name}' returned an error: {result.content!s}"
            raise MCPInvocationError(message)

        # Validate content types - only allow TextContent for now
        if result.content:
            for item in result.content:
                if not isinstance(item, types.TextContent):
                    # Reject any non-TextContent
                    message = (
                        f"Unsupported content type in response from tool '{tool_name}'. "
                        f"Only TextContent is currently supported."
                    )
                    raise MCPResponseTypeError(message)

        # Return the original result object
        return result

    async def close(self) -> None:
        """
        Close the connection and clean up resources.

        This method ensures all resources are properly released, even if errors occur.
        """
        if not self.exit_stack:
            return

        try:
            await self.exit_stack.aclose()
        except Exception as e:
            logger.warning(f"Error during MCP client cleanup: {e}")
        finally:
            # Ensure all references are cleared even if cleanup fails
            self.session = None
            self.stdio = None
            self.write = None

    async def _initialize_session_with_transport(
        self,
        transport_tuple: tuple[
            MemoryObjectReceiveStream[types.JSONRPCMessage | Exception], MemoryObjectSendStream[types.JSONRPCMessage]
        ],
        connection_type: str,
    ) -> list[Tool]:
        """
        Common session initialization logic for all transports.

        :param transport_tuple: Tuple containing (stdio, write) from the transport
        :param connection_type: String describing the connection type for error messages
        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        try:
            self.stdio, self.write = transport_tuple
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            # Now session is guaranteed to be a ClientSession, not None
            session = cast(ClientSession, self.session)  # Tell mypy the type is now known
            await session.initialize()

            # List available tools
            response = await session.list_tools()
            return response.tools

        except Exception as e:
            await self.close()
            message = f"Failed to connect to {connection_type}: {e}"
            raise MCPConnectionError(message) from e

    async def list_tools(self) -> list[Tool]:
        """
        List all available tools on the connected MCP server.

        :returns: List of available tools on the server
        :raises MCPError: If not connected to an MCP server
        """
        if not self.session:
            message = "Not connected to an MCP server"
            raise MCPError(message)

        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            message = f"Failed to list tools: {e}"
            raise MCPError(message) from e


class StdioMCPClient(MCPClient):
    """
    MCP client that connects to servers using stdio transport.
    """

    def __init__(self, command: str, args: list[str] | None = None, env: dict[str, str] | None = None) -> None:
        """
        Initialize a stdio MCP client.

        :param command: Command to run (e.g., "python", "node")
        :param args: Arguments to pass to the command
        :param env: Environment variables for the command
        """
        super().__init__()
        self.command: str = command
        self.args: list[str] = args or []
        self.env: dict[str, str] | None = env

    async def connect(self) -> list[Tool]:
        """
        Connect to an MCP server using stdio transport.

        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        server_params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        return await self._initialize_session_with_transport(stdio_transport, f"stdio server (command: {self.command})")


class HttpMCPClient(MCPClient):
    """
    MCP client that connects to servers using HTTP transport.
    """

    def __init__(self, base_url: str, token: str | None = None, timeout: int = 5) -> None:
        """
        Initialize an HTTP MCP client.

        :param base_url: Base URL of the server
        :param token: Authentication token for the server (optional)
        :param timeout: Connection timeout in seconds
        """
        super().__init__()
        self.base_url: str = base_url.rstrip("/")  # Remove any trailing slashes
        self.token: str | None = token
        self.timeout: int = timeout

    async def connect(self) -> list[Tool]:
        """
        Connect to an MCP server using HTTP transport.

        :returns: List of available tools on the server
        :raises MCPConnectionError: If connection to the server fails
        """
        sse_url = f"{self.base_url}/sse"
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(sse_url, headers=headers, timeout=self.timeout)
        )
        return await self._initialize_session_with_transport(sse_transport, f"HTTP server at {self.base_url}")


@dataclass
class MCPServerInfo(ABC):
    """
    Abstract base class for MCP server connection parameters.

    This class defines the common interface for all MCP server connection types.
    """

    @abstractmethod
    def create_client(self) -> MCPClient:
        """
        Create an appropriate MCP client for this server info.

        :returns: An instance of MCPClient configured with this server info
        """
        pass

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this server info to a dictionary.

        :returns: Dictionary representation of this server info
        """
        # Store the fully qualified class name for deserialization
        result = {"type": generate_qualified_class_name(type(self))}

        # Add all fields from the dataclass
        for field in fields(self):
            result[field.name] = getattr(self, field.name)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServerInfo":
        """
        Deserialize server info from a dictionary.

        :param data: Dictionary containing serialized server info
        :returns: Instance of the appropriate server info class
        """
        # Remove the type field as it's not a constructor parameter
        data_copy = data.copy()
        data_copy.pop("type", None)

        # Create an instance of the class with the remaining fields
        return cls(**data_copy)


@dataclass
class HttpMCPServerInfo(MCPServerInfo):
    """
    Data class that encapsulates HTTP MCP server connection parameters.

    :param base_url: Base URL of the MCP server
    :param token: Authentication token for the server (optional)
    :param timeout: Connection timeout in seconds
    """

    base_url: str
    token: str | None = None
    timeout: int = 30

    def create_client(self) -> MCPClient:
        """
        Create an HTTP MCP client.

        :returns: Configured HttpMCPClient instance
        """
        return HttpMCPClient(self.base_url, self.token, self.timeout)


@dataclass
class StdioMCPServerInfo(MCPServerInfo):
    """
    Data class that encapsulates stdio MCP server connection parameters.

    :param command: Command to run (e.g., "python", "node")
    :param args: Arguments to pass to the command
    :param env: Environment variables for the command
    """

    command: str
    args: list[str] | None = None
    env: dict[str, str] | None = None

    def create_client(self) -> MCPClient:
        """
        Create a stdio MCP client.

        :returns: Configured StdioMCPClient instance
        """
        return StdioMCPClient(self.command, self.args, self.env)


class MCPServer:
    """
    Represents a connection to an MCP server and manages available tools.
    
    Example using HTTP:
    ```python
    server = MCPServer(HttpMCPServerInfo(base_url="http://localhost:8000"))
    await server.connect()
    tools = server.available_tools
    add_tool = server.get_tool("add")
    result = await add_tool.invoke(a=5, b=3)
    ```

    Example using stdio:
    ```python
    server = MCPServer(StdioMCPServerInfo(command="python", args=["server.py"]))
    await server.connect()
    time_tool = server.get_tool("get_current_time")
    result = await time_tool.invoke(timezone="America/New_York")
    ```
    """

    def __init__(
        self,
        server_info: MCPServerInfo,
        connection_timeout: int = 30,
        invocation_timeout: int = 30,
    ):
        """Initialize connection to an MCP server."""
        self._server_info = server_info
        self._connection_timeout = connection_timeout
        self._invocation_timeout = invocation_timeout
        
        # Connect immediately and get tools
        self._client = self._server_info.create_client()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        tools = loop.run_until_complete(
            asyncio.wait_for(
                self._client.connect(),
                timeout=self._connection_timeout
            )
        )
        
        # Initialize tool instances
        self._available_tools = {
            tool_info.name: Tool(
                name=tool_info.name,
                description=tool_info.description,
                parameters=tool_info.inputSchema,
                function=self._create_tool_invoker(tool_info.name)
            ) for tool_info in tools
        }

    @property
    def available_tools(self) -> list[Tool]:
        """Get list of available tools on the server."""
        return list(self._available_tools.values())

    def get_tool(self, name: str) -> Tool:
        """Get a specific tool by name."""
        try:
            return self._available_tools[name]
        except KeyError:
            available = ", ".join(self._available_tools.keys())
            message = f"Tool '{name}' not found. Available tools: {available}"
            raise MCPToolNotFoundError(message)

    def _create_tool_invoker(self, tool_name: str):
        """Create a synchronous function that invokes a specific tool."""
        def invoke_tool(**kwargs):
            if not self._client:
                raise MCPError("Not connected to server")
            
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async call synchronously
            try:
                result = loop.run_until_complete(
                    asyncio.wait_for(
                        self._client.call_tool(tool_name, kwargs),
                        timeout=self._invocation_timeout
                    )
                )
                return self._process_result(result)
            except Exception as e:
                message = f"Error invoking tool '{tool_name}': {e}"
                raise MCPInvocationError(message) from e
                
        return invoke_tool

    def _process_result(self, result: types.CallToolResult) -> Any:
        """Process the result from a tool invocation."""
        if result.isError:
            if result.content and isinstance(result.content[0], types.TextContent):
                message = result.content[0].text
            else:
                message = str(result.content)
            raise MCPInvocationError(message)

        # For now, only handle TextContent
        if result.content:
            for item in result.content:
                if not isinstance(item, types.TextContent):
                    message = "Only TextContent is currently supported"
                    raise MCPResponseTypeError(message)
            
            # Return text content
            return [item.text for item in result.content]
        return None

    async def close(self) -> None:
        """Close the connection and clean up resources."""
        if self._client:
            try:
                # Get the current task's loop
                loop = asyncio.get_running_loop()
                # Only close if we're in the same loop that created the client
                if loop is asyncio.get_event_loop():
                    await self._client.close()
                else:
                    # We're in a different task, use run_until_complete
                    loop.run_until_complete(self._client.close())
            except Exception as e:
                logger.warning(f"Error during server cleanup: {e}")
            finally:
                self._client = None
                self._available_tools.clear()

