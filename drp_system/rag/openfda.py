"""Live drug label lookup from openFDA (no API key)."""
import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

BASE_URL = "https://api.fda.gov/drug/label.json"


@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
async def _fetch_label(client: httpx.AsyncClient, search: str) -> dict | None:
    resp = await client.get(BASE_URL, params={"search": search, "limit": 1})
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    if not data.get("results"):
        return None
    label = data["results"][0]
    return {
        "warnings": label.get("warnings", ["Not found"]),
        "contraindications": label.get("contraindications", ["Not found"]),
        "drug_interactions": label.get("drug_interactions", ["Not found"]),
        "indications_and_usage": label.get("indications_and_usage", ["Not found"]),
    }


async def get_drug_label(drug_name: str) -> dict:
    """Fetch FDA label by brand then generic name."""
    async with httpx.AsyncClient(timeout=15) as client:
        for field in ("openfda.brand_name", "openfda.generic_name"):
            result = await _fetch_label(
                client, f'{field}:"{drug_name}"'
            )
            if result:
                return result
    return {"error": f"No FDA label found for '{drug_name}'"}
