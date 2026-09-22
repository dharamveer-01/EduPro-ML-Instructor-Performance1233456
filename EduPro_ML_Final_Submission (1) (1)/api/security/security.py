
import os
from fastapi import Header, HTTPException


def get_configured_api_key():

    return os.getenv(
        "EDUPRO_API_KEY",
        "EDUPRO-DEMO-API-KEY-2026"
    )


async def verify_api_key(
    x_api_key: str = Header(default=None)
):

    configured_key = get_configured_api_key()

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key"
        )

    if x_api_key != configured_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return True
