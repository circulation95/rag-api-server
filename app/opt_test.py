import asyncio
from opc_client import IgnitionOpcClient

async def t():
    c = IgnitionOpcClient("opc.tcp://localhost:62541")
    print(await c.read_tag("[default]MEOMORY"))
    print(await c.write_tag("[default]MEOMORY", 1234))
    print(await c.read_tag("[default]MEOMORY"))
    await c.disconnect()

asyncio.run(t())