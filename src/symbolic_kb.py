import requests


WIKIDATA_API = "https://www.wikidata.org/w/api.php"

HEADERS = {
    "User-Agent": (
        "FEVER-Fact-Checking-Research/1.0 "
        "(Bowie State University student project)"
    )
}


def search_entity(entity_name, limit=5):
    """
    Resolve an entity name to Wikidata candidates.
    """

    params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "language": "en",
        "format": "json",
        "limit": limit,
    }

    try:
        response = requests.get(
            WIKIDATA_API,
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()

        results = []

        for item in response.json().get("search", []):
            results.append({
                "id": item.get("id"),
                "label": item.get("label"),
                "description": item.get(
                    "description",
                    "No description"
                ),
            })

        return results

    except requests.RequestException as error:
        print("Wikidata search failed:", error)
        return []


def get_entity_data(entity_id):
    """
    Retrieve structured Wikidata data for
    a resolved entity.
    """

    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "languages": "en",
        "props": "labels|descriptions|claims",
        "format": "json",
    }

    try:
        response = requests.get(
            WIKIDATA_API,
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()

        return (
            response.json()
            .get("entities", {})
            .get(entity_id, {})
        )

    except requests.RequestException as error:
        print("Wikidata entity lookup failed:", error)
        return {}


def symbolic_lookup(entity_name):
    """
    Resolve an entity and retrieve its
    structured Wikidata claims/properties.
    """

    matches = search_entity(
        entity_name,
        limit=1
    )

    if not matches:
        return {
            "resolved": False,
            "entity": entity_name,
            "entity_id": None,
            "properties": {},
        }

    best_match = matches[0]
    entity_id = best_match["id"]

    entity_data = get_entity_data(
        entity_id
    )

    properties = entity_data.get(
        "claims",
        {}
    )

    return {
        "resolved": True,
        "entity": entity_name,
        "entity_id": entity_id,
        "label": best_match["label"],
        "description": best_match["description"],
        "properties": properties,
    }


def main():

    print("\nWIKIDATA SYMBOLIC KB TEST")
    print("--------------------------------")

    entities = [
        "Nikolaj Coster-Waldau",
        "Fox Broadcasting Company",
    ]

    for entity_name in entities:

        result = symbolic_lookup(
            entity_name
        )

        print(
            f"\nEntity: {entity_name}"
        )

        if not result["resolved"]:
            print("Status: UNRESOLVED")
            continue

        print("Status: RESOLVED")

        print(
            "Wikidata ID:",
            result["entity_id"]
        )

        print(
            "Description:",
            result["description"]
        )

        print(
            "Number of property types:",
            len(result["properties"])
        )

        print(
            "Sample property IDs:",
            list(result["properties"].keys())[:10]
        )

    print(
        "\nStructured symbolic lookup passed."
    )


if __name__ == "__main__":
    main()