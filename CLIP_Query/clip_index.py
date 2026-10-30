import re
import base64
import numpy as np
import cv2
import chromadb
from chromadb.config import Settings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from CLIP_Query.clip_encoder import ClipEncoder


VEHICLE_TYPES = {0: 'motorcycle', 1: 'car', 2: 'truck', 3: 'bus'}

# keyword -> vehicle_type id (same ids as the YOLO model / ReID collections)
TYPE_KEYWORDS = {
    0: ["motorcycle", "motorcycles", "motorbike", "motorbikes", "scooter", "scooters", "bike", "moto"],
    1: ["car", "cars", "sedan", "suv", "hatchback", "taxi", "van"],
    2: ["truck", "trucks", "lorry", "pickup", "container"],
    3: ["bus", "buses", "coach"],
}

CAMERA_PATTERN = re.compile(r"\b(?:cam|camera)\s*_?(\d+)\b", re.IGNORECASE)

METADATA_KEYS = ['vehicle_id', 'vehicle_type', 'vehicle_type_name', 'confidence',
                 'camera_id', 'track_id', 'timestamp', 'image_encoded']


def make_doc_id(vehicle_type: int, vehicle_id: int, camera_id: str) -> str:
    """One CLIP entry per vehicle per camera, mirroring the ReID dedup rule."""
    return f"{int(vehicle_type)}_{int(vehicle_id)}_{camera_id}"


def encode_image(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def decode_image(encoded_image: str) -> np.ndarray:
    nparr = np.frombuffer(base64.b64decode(encoded_image), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def make_client(host: str = "localhost", port: int = 8000):
    return chromadb.HttpClient(
        host=host,
        port=port,
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )


def parse_query(query: str) -> Tuple[str, Optional[int], Optional[str]]:
    """Extract hard filters from free text.

    "white truck in cam2" -> ("white truck", 2, "cam2")
    The vehicle-type word is kept in the text because it also helps CLIP.
    """
    camera_id = None
    cam_match = CAMERA_PATTERN.search(query)
    if cam_match:
        camera_id = f"cam{cam_match.group(1)}"
        query = CAMERA_PATTERN.sub(" ", query)
        query = re.sub(r"\b(?:in|on|from|at)\s*$", " ", query.strip(), flags=re.IGNORECASE)

    vehicle_type = None
    words = set(re.findall(r"[a-z]+", query.lower()))
    for type_id, keywords in TYPE_KEYWORDS.items():
        if words.intersection(keywords):
            vehicle_type = type_id
            break

    clean = " ".join(query.split())
    if not clean:
        clean = VEHICLE_TYPES.get(vehicle_type, "vehicle")
    return clean, vehicle_type, camera_id


class ClipIndex:
    """Chroma collection of CLIP image embeddings, used for text -> vehicle search."""

    def __init__(self,
                 client,
                 encoder: "ClipEncoder",
                 collection_name: str = "vehicle_clip"):
        self.client = client
        self.encoder = encoder
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self.collection.count()

    def existing_ids(self, ids: List[str]) -> set:
        if not ids:
            return set()
        return set(self.collection.get(ids=ids, include=[])['ids'])

    def add_many(self, items: List[Dict]) -> int:
        """Index crops that are not indexed yet.

        Each item: {"id": str, "image": BGR ndarray, "metadata": dict}.
        Returns the number of newly added entries.
        """
        if not items:
            return 0
        # Deduplicate within the call, then skip ids already in the collection
        unique = {}
        for item in items:
            if item.get("image") is not None and item["image"].size > 0:
                unique.setdefault(item["id"], item)
        existing = self.existing_ids(list(unique.keys()))
        new_items = [item for doc_id, item in unique.items() if doc_id not in existing]
        if not new_items:
            return 0

        embeddings = self.encoder.encode_images([item["image"] for item in new_items])
        metadatas = []
        for item in new_items:
            metadata = self._clean_metadata(item.get("metadata", {}))
            if 'image_encoded' not in metadata:
                metadata['image_encoded'] = encode_image(item["image"])
            metadatas.append(metadata)

        self.collection.add(
            ids=[item["id"] for item in new_items],
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )
        return len(new_items)

    def search(self,
               query: str,
               top_k: int = 20,
               vehicle_type: Optional[int] = None,
               camera_id: Optional[str] = None,
               parse: bool = True) -> List[Dict]:
        """Return vehicles ranked by similarity to the text, grouped by ReID identity."""
        text = query
        if parse:
            text, parsed_type, parsed_cam = parse_query(query)
            vehicle_type = vehicle_type if vehicle_type is not None else parsed_type
            camera_id = camera_id or parsed_cam

        total = self.count()
        if total == 0:
            return []

        conditions = []
        if vehicle_type is not None:
            conditions.append({"vehicle_type": {"$eq": int(vehicle_type)}})
        if camera_id:
            conditions.append({"camera_id": {"$eq": camera_id}})
        where = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        text_embedding = self.encoder.encode_text(text)
        # Fetch extra hits so that grouping by vehicle still yields top_k vehicles
        n_results = min(total, top_k * 4)
        results = self.collection.query(
            query_embeddings=[text_embedding.tolist()],
            n_results=n_results,
            where=where,
            include=["metadatas", "distances"],
        )

        vehicles: Dict[Tuple[int, int], Dict] = {}
        for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
            score = 1.0 - distance
            key = (metadata.get('vehicle_type'), metadata.get('vehicle_id'))
            entry = vehicles.setdefault(key, {
                "vehicle_id": metadata.get('vehicle_id'),
                "vehicle_type": metadata.get('vehicle_type'),
                "vehicle_type_name": metadata.get('vehicle_type_name'),
                "score": score,
                "appearances": [],
            })
            entry["score"] = max(entry["score"], score)
            entry["appearances"].append({
                "camera_id": metadata.get('camera_id'),
                "track_id": metadata.get('track_id'),
                "timestamp": metadata.get('timestamp'),
                "score": score,
                "image_encoded": metadata.get('image_encoded'),
            })

        ranked = sorted(vehicles.values(), key=lambda v: v["score"], reverse=True)[:top_k]
        for vehicle in ranked:
            vehicle["appearances"].sort(key=lambda a: a["camera_id"] or "")
        return ranked

    @staticmethod
    def _clean_metadata(metadata: Dict) -> Dict:
        # Chroma only accepts str/int/float/bool values, never None
        cleaned = {}
        for key in METADATA_KEYS:
            value = metadata.get(key)
            if value is None:
                continue
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            cleaned[key] = value
        if 'vehicle_type' in cleaned and 'vehicle_type_name' not in cleaned:
            cleaned['vehicle_type_name'] = VEHICLE_TYPES.get(int(cleaned['vehicle_type']), 'unknown')
        return cleaned
