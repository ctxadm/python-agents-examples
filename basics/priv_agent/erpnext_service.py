"""
ERPNext Service für LiveKit Agent (async, analog EmailService).

Bietet:
  - search_customer / get_customer / create_customer / set_customer_contact
  - get_customer_email_from_invoice
  - find_item
  - create_quotation
  - create_invoice_draft / submit_invoice / send_invoice_email
  - get_open_invoices
"""
import os
import re
import json
import difflib
import logging
from typing import Optional
import httpx

logger = logging.getLogger("priv-agent")

ERPNEXT_URL = os.getenv("ERPNEXT_URL", "http://erpnext.fastlane-ai.ch:8080").rstrip("/")
ERPNEXT_API_KEY = os.getenv("ERPNEXT_API_KEY", "")
ERPNEXT_API_SECRET = os.getenv("ERPNEXT_API_SECRET", "")
ERPNEXT_COMPANY = os.getenv("ERPNEXT_COMPANY", "Fastlane AI GmbH")
ERPNEXT_CURRENCY = os.getenv("ERPNEXT_DEFAULT_CURRENCY", "CHF")
ERPNEXT_TAX_TEMPLATE = os.getenv("ERPNEXT_DEFAULT_TAX_TEMPLATE", "CH MwSt 8.1% - FAG")
ERPNEXT_INVOICE_THRESHOLD = float(os.getenv("ERPNEXT_INVOICE_AMOUNT_CONFIRM_THRESHOLD", "5000"))
ERPNEXT_TIMEOUT = float(os.getenv("ERPNEXT_TIMEOUT", "20"))

# Firmensuffixe für tolerantes Customer-Name-Matching
_COMPANY_SUFFIXES = (
    " ag", " gmbh", " sa", " sarl", " inc", " ltd", " llc",
    " gbr", " kg", " ohg", " e.k.", " e.v.", " ug",
)

# Mindest-Ähnlichkeits-Score, ab dem ein Fuzzy-Treffer als "exakt" gilt
_HIGH_CONFIDENCE_THRESHOLD = 0.85

# Minimum-Ähnlichkeit, ab der ein Customer überhaupt in die Fuzzy-Liste kommt
_MIN_FUZZY_SCORE = 0.4


def _normalize_customer_name(name: str) -> str:
    """Lowercase + gängigen Firmensuffix abschneiden für tolerantes Matching."""
    s = (name or "").lower().strip()
    for suf in _COMPANY_SUFFIXES:
        if s.endswith(suf):
            s = s[:-len(suf)].strip()
            break
    return s


def _normalize_phone_e164(phone: str, default_country_code: str = "+41") -> str:
    """
    Normalisiert eine Telefonnummer zu E.164-Format (CH-Default: +41).
    Beispiele:
      "044 123 45 67"     -> "+41441234567"
      "+41 44 123 45 67"  -> "+41441234567"
      "0041 44 123 45 67" -> "+41441234567"
      "+41441234567"      -> "+41441234567"
    """
    if not phone:
        return ""
    cleaned = re.sub(r"[^\d+]", "", phone.strip())
    if not cleaned:
        return ""
    if cleaned.startswith("+"):
        return cleaned
    if cleaned.startswith("00"):
        return "+" + cleaned[2:]
    if cleaned.startswith("0"):
        return default_country_code + cleaned[1:]
    return default_country_code + cleaned


def _is_valid_email(value) -> bool:
    """
    Defensiver Filter gegen LLM-Pannen: akzeptiert nur echte Email-Strings.
    Filtert Platzhalter wie 'None', 'null', leere Strings, Whitespace, fehlendes '@'.
    """
    if not value or not isinstance(value, str):
        return False
    s = value.strip()
    if not s:
        return False
    if s.lower() in ("none", "null", "n/a", "na", "-"):
        return False
    if "@" not in s or "." not in s:
        return False
    return True


class ERPNextService:
    """Async REST API Client für ERPNext v16."""

    def __init__(self):
        self.base_url = ERPNEXT_URL
        self.company = ERPNEXT_COMPANY
        self.currency = ERPNEXT_CURRENCY
        self.tax_template = ERPNEXT_TAX_TEMPLATE
        self.invoice_threshold = ERPNEXT_INVOICE_THRESHOLD
        self._client: Optional[httpx.AsyncClient] = None
        logger.info(f"🔗 ERPNext Service: {self.base_url} / {self.company}")

    async def start(self):
        if not (ERPNEXT_API_KEY and ERPNEXT_API_SECRET):
            logger.warning("⚠️  ERPNEXT_API_KEY/SECRET nicht gesetzt")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"token {ERPNEXT_API_KEY}:{ERPNEXT_API_SECRET}",
                "Content-Type": "application/json",
            },
            timeout=ERPNEXT_TIMEOUT,
        )

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, path: str, **kwargs) -> tuple[bool, dict | str]:
        if not self._client:
            await self.start()
        try:
            r = await self._client.request(method, path, **kwargs)
            if r.status_code >= 400:
                logger.error(f"❌ ERPNext {method} {path} → {r.status_code}: {r.text[:300]}")
                if r.status_code in (401, 403):
                    return False, "Zugriff auf ERPNext ist nicht möglich."
                if r.status_code == 404:
                    return False, "Das angeforderte Dokument existiert nicht in ERPNext."
                return False, f"ERPNext meldete einen Fehler ({r.status_code})."
            return True, r.json()
        except httpx.TimeoutException:
            logger.error(f"❌ ERPNext Timeout: {method} {path}")
            return False, "ERPNext antwortet im Moment nicht."
        except Exception as e:
            logger.error(f"❌ ERPNext Fehler: {e}")
            return False, "Ein Fehler ist bei ERPNext aufgetreten."

    # ========================================================================
    # CUSTOMER
    # ========================================================================

    async def search_customer(self, query: str) -> tuple[bool, dict | str]:
        """
        Suche Customers mit mehrstufiger Strategie:
          1. Exact (case-insensitive + Firmensuffix-tolerant)
          2. Substring-Präfix-Suche (4-Zeichen-Tokens, schnell)
          2b. Fetch-All Fallback wenn Stufe 2 leer (typo-tolerant an jeder Position)
          3. Re-Ranking via difflib.SequenceMatcher;
             Top-1 mit Score >= 0.85 wird als exact_match gewertet.

        Returns:
            (True, {"exact_match": bool, "results": list[dict]})
        """
        query_clean = (query or "").strip()
        if not query_clean:
            return True, {"exact_match": False, "results": []}
        query_norm = _normalize_customer_name(query_clean)

        # === Stufe 1: Case-insensitive Substring + Suffix-tolerantes Match ===
        ok, data = await self._request("GET", "/api/resource/Customer", params={
            "filters": json.dumps([["customer_name", "like", f"%{query_clean}%"]]),
            "fields": json.dumps(["name", "customer_name", "customer_group"]),
            "limit_page_length": 20,
        })
        if not ok:
            return False, data

        candidates = data.get("data", [])
        exact_matches = [
            c for c in candidates
            if _normalize_customer_name(c["customer_name"]) == query_norm
        ]
        if len(exact_matches) == 1:
            logger.info(f"   🎯 Exact-Match (case/suffix-tolerant): {exact_matches[0]['customer_name']}")
            return True, {"exact_match": True, "results": exact_matches}

        # === Stufe 2: Substring-Präfix-Suche (typo-tolerant am Wortende) ===
        tokens = [t for t in re.split(r"[\s\-_.]+", query_clean) if len(t) >= 3]
        if not tokens:
            tokens = [query_clean]
        short_tokens = [t[:4] if len(t) > 4 else t for t in tokens]
        or_filters = [["customer_name", "like", f"%{t}%"] for t in short_tokens]

        ok, data = await self._request("GET", "/api/resource/Customer", params={
            "or_filters": json.dumps(or_filters),
            "fields": json.dumps(["name", "customer_name", "customer_group"]),
            "limit_page_length": 50,
        })
        if not ok:
            return False, data
        raw_results = data.get("data", [])

        # === Stufe 2b: Fetch-All Fallback (typo-tolerant an JEDER Position) ===
        if not raw_results:
            logger.info(f"   ⚠️ Stufe 2 leer für '{query_clean}' – Fallback Fetch-All")
            ok, data = await self._request("GET", "/api/resource/Customer", params={
                "fields": json.dumps(["name", "customer_name", "customer_group"]),
                "limit_page_length": 0,
            })
            if not ok:
                return False, data
            raw_results = data.get("data", [])
            logger.info(f"   📥 Fetch-All: {len(raw_results)} Customers geladen")

        if not raw_results:
            logger.info(f"   ❌ Kein Treffer für '{query_clean}'")
            return True, {"exact_match": False, "results": []}

        # === Stufe 3: Re-Ranking via SequenceMatcher ===
        scored = [
            (
                difflib.SequenceMatcher(
                    None,
                    query_norm,
                    _normalize_customer_name(c["customer_name"]),
                ).ratio(),
                c,
            )
            for c in raw_results
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top_customer = scored[0]
        logger.info(
            f"   🔍 Top-Score: {top_customer['customer_name']} = {top_score:.2f} "
            f"(Query: '{query_clean}')"
        )

        if top_score < _MIN_FUZZY_SCORE:
            logger.info(f"   ❌ Top-Score {top_score:.2f} unter Schwelle {_MIN_FUZZY_SCORE}")
            return True, {"exact_match": False, "results": []}

        if top_score >= _HIGH_CONFIDENCE_THRESHOLD:
            logger.info(f"   🎯 High-Confidence-Match: {top_customer['customer_name']}")
            return True, {"exact_match": True, "results": [top_customer]}

        return True, {
            "exact_match": False,
            "results": [c for s, c in scored[:5] if s >= _MIN_FUZZY_SCORE],
        }

    async def get_customer(self, name: str) -> tuple[bool, dict | str]:
        """
        Liest Customer-Details inkl. Email/Phone mit 3-stufigem Fallback.
        Liefert zusätzlich address_line1, pincode und city einzeln,
        damit die Adresse im Tool-Layer Voice-/Chat-konform formatiert werden kann.
        """
        ok, data = await self._request("GET", f"/api/resource/Customer/{name}")
        if not ok:
            return False, data
        cust = data.get("data", {})
        details = {
            "name": cust.get("name"),
            "customer_name": cust.get("customer_name"),
            "customer_group": cust.get("customer_group"),
            "email": None,
            "phone": None,
            "address": None,         # combined (kompatibel)
            "address_line1": None,   # neu: Strasse separat
            "pincode": None,         # neu: PLZ separat
            "city": None,            # neu: Ort separat
        }

        # --- Stufe 1: Direktes email_id-Feld am Customer ---
        if cust.get("email_id"):
            details["email"] = cust["email_id"]
            logger.info(f"   📧 Email aus Customer.email_id: {details['email']}")

        # --- Stufe 2: customer_primary_contact (klassischer Weg) ---
        if cust.get("customer_primary_contact"):
            ok2, contact = await self._request(
                "GET", f"/api/resource/Contact/{cust['customer_primary_contact']}"
            )
            if ok2:
                c = contact.get("data", {})
                if not details["email"] and c.get("email_ids"):
                    details["email"] = c["email_ids"][0].get("email_id")
                    logger.info(f"   📧 Email aus Primary Contact: {details['email']}")
                if c.get("phone_nos"):
                    details["phone"] = c["phone_nos"][0].get("phone")

        # --- Stufe 3: Fallback – Contacts via Dynamic Link suchen ---
        if not details["email"] or not details["phone"]:
            ok3, contacts_data = await self._request(
                "GET", "/api/resource/Contact",
                params={
                    "filters": json.dumps([
                        ["Dynamic Link", "link_doctype", "=", "Customer"],
                        ["Dynamic Link", "link_name", "=", cust.get("name")],
                    ]),
                    "fields": json.dumps(["name", "email_id", "phone"]),
                    "limit_page_length": 10,
                },
            )
            if ok3:
                for contact in contacts_data.get("data", []):
                    if not details["email"] and contact.get("email_id"):
                        details["email"] = contact["email_id"]
                        logger.info(
                            f"   📧 Email via Dynamic-Link-Fallback aus '{contact['name']}': {details['email']}"
                        )
                    if not details["phone"] and contact.get("phone"):
                        details["phone"] = contact["phone"]
                    if details["email"] and details["phone"]:
                        break

        # --- Adresse (jetzt mit Einzelfeldern) ---
        if cust.get("customer_primary_address"):
            ok4, addr = await self._request(
                "GET", f"/api/resource/Address/{cust['customer_primary_address']}"
            )
            if ok4:
                a = addr.get("data", {})
                details["address_line1"] = a.get("address_line1") or None
                details["pincode"] = a.get("pincode") or None
                details["city"] = a.get("city") or None
                # Kombiniertes Feld für Backwards-Compat
                line1 = a.get("address_line1", "")
                pin = a.get("pincode", "")
                city = a.get("city", "")
                combined = f"{line1}, {pin} {city}".strip(", ").strip()
                details["address"] = combined or None

        return True, details

    async def get_customer_email_from_invoice(self, invoice_name: str) -> Optional[str]:
        """Holt die primäre Email des Customers, der zu einer Invoice gehört."""
        ok, data = await self._request("GET", f"/api/resource/Sales Invoice/{invoice_name}")
        if not ok:
            return None
        customer_id = data.get("data", {}).get("customer") if isinstance(data, dict) else None
        if not customer_id:
            return None
        ok2, details = await self.get_customer(customer_id)
        if not ok2 or not isinstance(details, dict):
            return None
        return details.get("email")

    async def create_customer(
        self,
        customer_name: str,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        customer_group: str = "Commercial",
    ) -> tuple[bool, str]:
        ok, results = await self.search_customer(customer_name)
        if ok and isinstance(results, dict) and results.get("exact_match"):
            existing = results["results"][0]
            return False, f"Kunde existiert bereits: {existing['name']}"

        ok, data = await self._request("POST", "/api/resource/Customer", json={
            "customer_name": customer_name,
            "customer_type": "Company",
            "customer_group": customer_group,
            "territory": "Switzerland",
        })
        if not ok:
            return False, data
        customer_id = data["data"]["name"]
        logger.info(f"✅ Customer angelegt: {customer_id}")

        # Defensiv: Email/Phone nur akzeptieren wenn echte Werte
        email_valid = _is_valid_email(email)
        phone_clean = _normalize_phone_e164(phone) if phone else ""

        if email_valid or phone_clean:
            payload = {
                "first_name": customer_name[:140],
                "links": [{"link_doctype": "Customer", "link_name": customer_id}],
            }
            if email_valid:
                payload["email_ids"] = [{"email_id": email.strip(), "is_primary": 1}]
            if phone_clean:
                payload["phone_nos"] = [{
                    "phone": phone_clean,
                    "is_primary_phone": 1,
                    "is_primary_mobile_no": 1,
                }]

            ok2, contact_data = await self._request("POST", "/api/resource/Contact", json=payload)
            if ok2:
                contact_id = contact_data["data"]["name"]
                await self._request("PUT", f"/api/resource/Customer/{customer_id}",
                                    json={"customer_primary_contact": contact_id})
                logger.info(f"   Contact verknüpft: {contact_id}")
            else:
                logger.warning(f"   ⚠️ Contact-Anlage fehlgeschlagen: {contact_data}")

        return True, customer_id

    async def set_customer_contact(
        self,
        customer_id: str,
        phone: Optional[str] = None,
        email: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Setzt oder ergänzt den primären Kontakt eines bestehenden Kunden.

        Verhalten (Sanftes Phase 2):
          - Wenn der Kunde KEINEN primären Kontakt hat:
              → Neuen Contact anlegen mit phone und/oder email, dann verlinken
          - Wenn ein primärer Kontakt existiert, dort aber phone ODER email LEER ist:
              → Leeres Feld wird ergänzt (Merge), niemals überschreiben
          - Wenn ein Feld bereits gesetzt ist und neu beschrieben werden soll:
              → Ablehnen mit klarer Meldung (Schutz vor Datenverlust)

        Mindestens einer der Parameter phone oder email muss übergeben werden.
        Telefon wird zu E.164 normalisiert (CH-Default: +41).
        Email wird defensiv geprüft (Platzhalter werden ignoriert).

        Args:
            customer_id: Customer-ID aus ERPNext (exakter Name)
            phone: Telefonnummer (optional)
            email: E-Mail-Adresse (optional)

        Returns:
            (True, statusmeldung) bei Erfolg
            (False, fehlermeldung) bei Konflikt oder Fehler
        """
        # 0) Eingangs-Validierung
        phone_e164 = _normalize_phone_e164(phone) if phone else ""
        email_valid = _is_valid_email(email)
        email_clean = email.strip() if email_valid else ""

        if not phone_e164 and not email_valid:
            return False, (
                "Es muss mindestens eine gültige Telefonnummer oder E-Mail-Adresse "
                "übergeben werden."
            )

        # 1) Customer lesen → primary_contact prüfen
        ok, data = await self._request("GET", f"/api/resource/Customer/{customer_id}")
        if not ok:
            return False, data if isinstance(data, str) else "Kunde nicht gefunden."

        cust = data.get("data", {})
        customer_name = cust.get("customer_name") or customer_id
        primary_contact_id = cust.get("customer_primary_contact")

        # =====================================================================
        # FALL A: KEIN primary_contact → neuen Contact anlegen
        # =====================================================================
        if not primary_contact_id:
            return await self._create_and_link_new_contact(
                customer_id, customer_name, phone_e164, email_clean
            )

        # =====================================================================
        # FALL B: primary_contact existiert → Merge-Logik
        # =====================================================================
        ok, contact_resp = await self._request(
            "GET", f"/api/resource/Contact/{primary_contact_id}"
        )
        if not ok:
            return False, (
                f"Primärer Kontakt {primary_contact_id} konnte nicht geladen werden."
            )

        contact = contact_resp.get("data", {}) if isinstance(contact_resp, dict) else {}
        existing_phones = contact.get("phone_nos") or []
        existing_emails = contact.get("email_ids") or []

        has_phone = len(existing_phones) > 0
        has_email = len(existing_emails) > 0

        # Konflikt-Erkennung
        phone_conflict = bool(phone_e164) and has_phone
        email_conflict = bool(email_valid) and has_email

        # Wenn ALLE angefragten Felder bereits belegt sind → ablehnen
        wants_phone = bool(phone_e164)
        wants_email = bool(email_valid)

        if wants_phone and wants_email and phone_conflict and email_conflict:
            return False, (
                f"Beim Kunden {customer_name} sind bereits Telefonnummer und E-Mail "
                f"hinterlegt. Das Überschreiben bestehender Daten ist nicht freigegeben. "
                f"Bitte im ERPNext-UI manuell ändern."
            )
        if wants_phone and not wants_email and phone_conflict:
            return False, (
                f"Beim Kunden {customer_name} ist bereits eine Telefonnummer hinterlegt. "
                f"Das Überschreiben ist nicht freigegeben. Bitte im ERPNext-UI ändern."
            )
        if wants_email and not wants_phone and email_conflict:
            return False, (
                f"Beim Kunden {customer_name} ist bereits eine E-Mail-Adresse hinterlegt. "
                f"Das Überschreiben ist nicht freigegeben. Bitte im ERPNext-UI ändern."
            )

        # Merge: bestehende Listen 1:1 übernehmen + neue Einträge nur in leere Slots
        merged_phones = list(existing_phones)
        merged_emails = list(existing_emails)
        added_phone = False
        added_email = False

        if wants_phone and not has_phone:
            merged_phones.append({
                "phone": phone_e164,
                "is_primary_phone": 1,
                "is_primary_mobile_no": 1,
            })
            added_phone = True

        if wants_email and not has_email:
            merged_emails.append({
                "email_id": email_clean,
                "is_primary": 1,
            })
            added_email = True

        if not added_phone and not added_email:
            # Sollte durch Konflikt-Prüfung oben abgefangen sein – Safety-Net
            return False, (
                f"Keine Änderung möglich: alle angefragten Felder sind bereits gesetzt."
            )

        # PUT Contact: nur die geänderten Listen mitschicken
        update_payload = {}
        if added_phone:
            update_payload["phone_nos"] = merged_phones
        if added_email:
            update_payload["email_ids"] = merged_emails

        ok, put_result = await self._request(
            "PUT",
            f"/api/resource/Contact/{primary_contact_id}",
            json=update_payload,
        )
        if not ok:
            return False, (
                f"Die Aktualisierung des Kontakts schlug fehl: {put_result}"
            )

        logger.info(
            f"✅ Contact {primary_contact_id} ergänzt: "
            f"phone_added={added_phone}, email_added={added_email}"
        )

        # Status-Meldung
        if added_phone and added_email:
            return True, f"Telefonnummer und E-Mail wurden bei {customer_name} ergänzt."
        if added_phone:
            return True, f"Telefonnummer wurde bei {customer_name} ergänzt."
        return True, f"E-Mail-Adresse wurde bei {customer_name} ergänzt."

    async def _create_and_link_new_contact(
        self,
        customer_id: str,
        customer_name: str,
        phone_e164: str,
        email_clean: str,
    ) -> tuple[bool, str]:
        """
        Hilfsmethode: legt einen neuen Contact an und verlinkt ihn als
        customer_primary_contact. Verwendet von set_customer_contact (Fall A).
        """
        payload = {
            "first_name": customer_name[:140],
            "links": [{"link_doctype": "Customer", "link_name": customer_id}],
        }
        if phone_e164:
            payload["phone_nos"] = [{
                "phone": phone_e164,
                "is_primary_phone": 1,
                "is_primary_mobile_no": 1,
            }]
        if email_clean:
            payload["email_ids"] = [{
                "email_id": email_clean,
                "is_primary": 1,
            }]

        ok, contact_data = await self._request(
            "POST", "/api/resource/Contact", json=payload
        )
        if not ok:
            return False, contact_data
        contact_id = contact_data["data"]["name"]
        logger.info(f"✅ Contact NEU angelegt: {contact_id} für {customer_id}")

        ok, link_data = await self._request(
            "PUT",
            f"/api/resource/Customer/{customer_id}",
            json={"customer_primary_contact": contact_id},
        )
        if not ok:
            return False, (
                f"Kontakt {contact_id} angelegt, aber Verknüpfung mit Kunde "
                f"fehlgeschlagen: {link_data}"
            )

        logger.info(f"   ✅ Customer {customer_id} → Primary Contact {contact_id}")

        if phone_e164 and email_clean:
            return True, f"Telefonnummer und E-Mail wurden für {customer_name} angelegt."
        if phone_e164:
            return True, f"Telefonnummer wurde für {customer_name} angelegt."
        return True, f"E-Mail-Adresse wurde für {customer_name} angelegt."

    # ========================================================================
    # ITEM
    # ========================================================================

    async def find_item(self, query: str) -> tuple[bool, list[dict] | str]:
        ok, data = await self._request("GET", "/api/resource/Item", params={
            "filters": json.dumps([
                ["item_name", "like", f"%{query}%"],
                ["disabled", "=", 0],
            ]),
            "fields": json.dumps(["item_code", "item_name", "stock_uom", "standard_rate"]),
            "limit_page_length": 10,
        })
        if not ok:
            return False, data
        return True, data.get("data", [])

    # ========================================================================
    # QUOTATION
    # ========================================================================

    async def create_quotation(self, customer_id: str, items: list[dict]) -> tuple[bool, dict | str]:
        ok, data = await self._request("POST", "/api/resource/Quotation", json={
            "party_name": customer_id,
            "quotation_to": "Customer",
            "company": self.company,
            "currency": self.currency,
            "items": items,
            "taxes_and_charges": self.tax_template,
        })
        if not ok:
            return False, data
        q = data["data"]
        return True, {
            "name": q["name"],
            "grand_total": q.get("grand_total", 0),
            "currency": q.get("currency", self.currency),
        }

    # ========================================================================
    # SALES INVOICE
    # ========================================================================

    async def create_invoice_draft(self, customer_id: str, items: list[dict]) -> tuple[bool, dict | str]:
        ok, data = await self._request("POST", "/api/resource/Sales Invoice", json={
            "customer": customer_id,
            "company": self.company,
            "currency": self.currency,
            "items": items,
            "taxes_and_charges": self.tax_template,
        })
        if not ok:
            return False, data
        inv = data["data"]
        return True, {
            "name": inv["name"],
            "grand_total": inv.get("grand_total", 0),
            "currency": inv.get("currency", self.currency),
            "items_count": len(inv.get("items", [])),
        }

    async def submit_invoice(self, invoice_name: str) -> tuple[bool, str]:
        ok, data = await self._request("GET", f"/api/resource/Sales Invoice/{invoice_name}")
        if not ok:
            if isinstance(data, str) and "existiert nicht" in data:
                return False, (
                    f"Rechnung '{invoice_name}' existiert nicht in ERPNext. "
                    f"Wahrscheinlich wurde kein Rechnungsentwurf erstellt. "
                    f"Rufe zuerst erp_create_invoice_draft auf und verwende den dort "
                    f"zurückgegebenen Namen."
                )
            return False, data
        doc = data.get("data", {})
        ok2, result = await self._request("POST", "/api/method/frappe.client.submit",
                                          json={"doc": json.dumps(doc)})
        if not ok2:
            return False, result
        logger.info(f"✅ Invoice submitted: {invoice_name}")
        return True, invoice_name

    async def send_invoice_email(self, invoice_name: str, recipient: str) -> tuple[bool, str]:
        ok, data = await self._request(
            "POST",
            "/api/method/frappe.core.doctype.communication.email.make",
            json={
                "doctype": "Sales Invoice",
                "name": invoice_name,
                "subject": f"Rechnung {invoice_name} - {self.company}",
                "content": (
                    f"<p>Sehr geehrte Damen und Herren,</p>"
                    f"<p>im Anhang erhalten Sie unsere Rechnung <b>{invoice_name}</b>.</p>"
                    f"<p>Zahlbar innert 30 Tagen.</p>"
                    f"<p>Freundliche Grüsse<br>{self.company}</p>"
                ),
                "recipients": recipient,
                "send_email": 1,
                "print_format": "Standard",
                "attach_document_print": 1,
                "communication_medium": "Email",
            },
        )
        if not ok:
            return False, data
        logger.info(f"✅ Invoice email an {recipient}: {invoice_name}")
        return True, recipient

    # ========================================================================
    # READ-ONLY HELPER
    # ========================================================================

    async def get_open_invoices(self, customer_id: str) -> tuple[bool, list[dict] | str]:
        ok, data = await self._request("GET", "/api/resource/Sales Invoice", params={
            "filters": json.dumps([
                ["customer", "=", customer_id],
                ["status", "in", ["Unpaid", "Overdue", "Partly Paid"]],
                ["docstatus", "=", 1],
            ]),
            "fields": json.dumps(["name", "grand_total", "outstanding_amount", "due_date", "status"]),
            "limit_page_length": 20,
        })
        if not ok:
            return False, data
        return True, data.get("data", [])
