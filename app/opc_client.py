import asyncio
import logging
from typing import Any, Optional, Sequence
from asyncua import Client, ua

logger = logging.getLogger(__name__)


class IgnitionOpcClient:
    """
    Ignition OPC UA Server (Security=None, Anonymous) 전용 클라이언트
    - Username/Password/인증서 사용 안 함
    - 연결 유지 + 끊기면 재연결(backoff)
    """

    def __init__(
        self,
        endpoint_url: str = "opc.tcp://localhost:62541",
        namespace_index: int = 2,
        reconnect_backoff: Sequence[float] = (0.5, 1.0, 2.0, 3.0, 5.0),
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.namespace_index = namespace_index
        self.reconnect_backoff = tuple(reconnect_backoff)

        self._client: Optional[Client] = None
        self._connected: bool = False
        self._lock = asyncio.Lock()

    # -------------------------
    # Helpers
    # -------------------------
    def _normalize_tag_path(self, tag_path: str) -> str:
        # [default]TEST  -> [default]/TEST
        if "]" in tag_path and "]/" not in tag_path:
            tag_path = tag_path.replace("]", "]/", 1)
        return tag_path

    def _node_id(self, tag_path: str) -> str:
        tag_path = self._normalize_tag_path(tag_path)
        return f"ns={self.namespace_index};s={tag_path}"

    async def _connect_once(self):
        client = Client(url=self.endpoint_url)
        # ✅ Security=None / Anonymous (아무 설정도 하지 않음)
        await client.connect()
        self._client = client
        self._connected = True
        logger.info("✅ OPC UA connected (Anonymous / Security=None)")

    async def _connect_with_retries(self):
        last_err: Optional[Exception] = None
        for delay in (0.0, *self.reconnect_backoff):
            if delay:
                await asyncio.sleep(delay)
            try:
                await self._connect_once()
                return
            except Exception as e:
                last_err = e
                logger.warning("OPC UA connect failed (will retry): %s", e)

        raise RuntimeError(f"Failed to connect to OPC UA server: {last_err}") from last_err

    # -------------------------
    # Public
    # -------------------------
    async def connect(self):
        async with self._lock:
            if self._connected and self._client:
                return
            await self._connect_with_retries()

    async def disconnect(self):
        async with self._lock:
            if self._client:
                try:
                    await self._client.disconnect()
                finally:
                    self._client = None
                    self._connected = False
                    logger.info("🔌 OPC UA disconnected")

    async def _ensure(self):
        if not (self._connected and self._client):
            await self.connect()

    async def read_tag(self, tag_path: str) -> dict:
        await self._ensure()
        node_id = self._node_id(tag_path)

        try:
            node = self._client.get_node(node_id)
            dv = await node.read_data_value()

            return {
                "tag": tag_path,
                "nodeId": node_id,
                "value": dv.Value.Value,
                "status": dv.StatusCode.name,
            }

        except Exception as e:
            # 끊김이면 다음 호출에서 자동 재연결되도록 상태를 내려둠
            async with self._lock:
                self._connected = False
                self._client = None
            return {"tag": tag_path, "nodeId": node_id, "error": str(e)}

    async def write_tag(self, tag_path: str, value: Any) -> dict:
        await self._ensure()
        node_id = self._node_id(tag_path)

        try:
            node = self._client.get_node(node_id)

            # 타입 맞춰서 쓰기 (VariantType 유지)
            dv = await node.read_data_value()
            vtype = dv.Value.VariantType

            # 필요 시 기본 캐스팅 (문자 -> 숫자)
            cur = dv.Value.Value
            if isinstance(cur, bool):
                if isinstance(value, str):
                    value = value.strip().lower() in ("1", "true", "yes", "on")
                else:
                    value = bool(value)
            elif isinstance(cur, int) and not isinstance(value, int):
                value = int(value)
            elif isinstance(cur, float) and not isinstance(value, float):
                value = float(value)

            await node.write_value(ua.Variant(value, vtype))

            return {"tag": tag_path, "nodeId": node_id, "written": value, "status": "OK"}

        except Exception as e:
            async with self._lock:
                self._connected = False
                self._client = None
            return {"tag": tag_path, "nodeId": node_id, "error": str(e)}
