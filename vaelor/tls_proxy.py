"""Small bounded TLS terminator for the loopback-only noVNC gateway."""

from __future__ import annotations

import asyncio
import ssl

from .runtime_paths import app_path, env_value


LISTEN_HOST = env_value(
    "VAELOR_TLS_PROXY_HOST", "PM_TLS_PROXY_HOST", "0.0.0.0"
)
LISTEN_PORT = int(env_value(
    "VAELOR_TLS_PROXY_PORT", "PM_TLS_PROXY_PORT", "34002"
))
TARGET_HOST = env_value(
    "VAELOR_TLS_PROXY_TARGET_HOST", "PM_TLS_PROXY_TARGET_HOST", "127.0.0.1"
)
TARGET_PORT = int(env_value(
    "VAELOR_TLS_PROXY_TARGET_PORT", "PM_TLS_PROXY_TARGET_PORT", "34003"
))
MAX_CONNECTIONS = max(1, min(int(env_value(
    "VAELOR_TLS_PROXY_MAX_CONNECTIONS", "PM_TLS_PROXY_MAX_CONNECTIONS", "32"
)), 128))
BUFFER_SIZE = 64 * 1024


class TlsProxy:
    def __init__(self):
        self.slots = asyncio.Semaphore(MAX_CONNECTIONS)

    @staticmethod
    async def _copy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            while True:
                data = await reader.read(BUFFER_SIZE)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, RuntimeError):
                pass

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        if self.slots.locked():
            writer.close()
            await writer.wait_closed()
            return
        async with self.slots:
            try:
                target_reader, target_writer = await asyncio.wait_for(
                    asyncio.open_connection(TARGET_HOST, TARGET_PORT),
                    timeout=5,
                )
            except (OSError, asyncio.TimeoutError):
                writer.close()
                await writer.wait_closed()
                return
            client_to_target = asyncio.create_task(
                self._copy(reader, target_writer)
            )
            target_to_client = asyncio.create_task(
                self._copy(target_reader, writer)
            )
            done, pending = await asyncio.wait(
                {client_to_target, target_to_client},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)


async def serve():
    cert = env_value(
        "VAELOR_TLS_CERT", "PM_TLS_CERT", app_path("tls/vaelor.crt")
    )
    key = env_value(
        "VAELOR_TLS_KEY", "PM_TLS_KEY", app_path("tls/vaelor.key")
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert, key)
    proxy = TlsProxy()
    server = await asyncio.start_server(
        proxy.handle,
        LISTEN_HOST,
        LISTEN_PORT,
        ssl=context,
        backlog=64,
    )
    async with server:
        await server.serve_forever()


def main():
    asyncio.run(serve())


if __name__ == "__main__":
    main()
