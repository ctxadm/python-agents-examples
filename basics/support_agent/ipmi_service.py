# File: basics/support_agent/ipmi_service.py
# IPMI Power Service Client - kommuniziert mit dem FastAPI-Service hinter dem
# NetScaler-VIP. Pflicht-Konfiguration via ENV (kein Default für URL/Token).

import logging
import os
import httpx

logger = logging.getLogger("ipmi-service")
logger.setLevel(logging.INFO)


class IPMIService:
    def __init__(self):
        # Pflicht-ENVs (kein Default für sensible Daten / Infrastruktur-URLs)
        try:
            self.base_url = os.environ["IPMI_API_URL"].rstrip("/")
            self.token = os.environ["IPMI_API_TOKEN"]
        except KeyError as e:
            raise RuntimeError(f"Pflicht-ENV {e} fehlt für IPMIService") from None

        self.timeout = int(os.getenv("IPMI_TIMEOUT", "10"))
        self._client: httpx.AsyncClient | None = None
        logger.info(f"⚡ IPMI Service: {self.base_url}")

    async def start(self):
        """Initialisiert den HTTP-Client (analog zu ERPNextService.start)."""
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={
                "X-API-Token": self.token,
                "Content-Type": "application/json",
            },
        )

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # =========================================================================
    # INTERNE HELPER
    # =========================================================================

    async def _request_power(self, server: str, action: str) -> tuple[bool, dict | str]:
        """Sendet POST /power und gibt (ok, payload|error_message) zurück."""
        if not self._client:
            return False, "IPMI-Client nicht initialisiert"

        try:
            r = await self._client.post(
                f"{self.base_url}/power",
                json={"server": server, "action": action},
            )
        except httpx.TimeoutException:
            logger.error(f"❌ Timeout {server}/{action}")
            return False, "Timeout: IPMI-Service antwortet nicht"
        except httpx.RequestError as e:
            logger.error(f"❌ Verbindungsfehler {server}/{action}: {e}")
            return False, f"Verbindungsfehler: {e}"

        if r.status_code == 200:
            return True, r.json()

        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text or f"HTTP {r.status_code}"
        logger.warning(f"⚠️  {server}/{action} → HTTP {r.status_code}: {detail}")
        return False, str(detail)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def power_on(self, server: str) -> tuple[bool, str]:
        """
        Schaltet einen Server ein (chassis power on).
        Rückgabe: (ok, message)
        """
        ok, result = await self._request_power(server, "on")
        if not ok:
            return False, str(result)
        msg = result.get("message", "") if isinstance(result, dict) else str(result)
        return True, msg

    async def power_status(self, server: str) -> tuple[bool, str]:
        """
        Liest den aktuellen Power-Status (chassis power status).
        Rückgabe: (ok, message wie 'Chassis Power is on' / 'Chassis Power is off')
        """
        ok, result = await self._request_power(server, "status")
        if not ok:
            return False, str(result)
        msg = result.get("message", "") if isinstance(result, dict) else str(result)
        return True, msg
