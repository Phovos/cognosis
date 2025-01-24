import hashlib
import toml
from typing import List, Dict

class MerkleNode:
    """Represents a node in the Merkle ring."""
    def __init__(self, data: str):
        self.data = data
        self.hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Computes the hash for the node's data."""
        return hashlib.sha256(self.data.encode()).hexdigest()

class MerkleRing:
    """Represents a Merkle ring using TOML as the storage format."""
    def __init__(self):
        self.nodes: List[MerkleNode] = []

    def add_node(self, data: str):
        """Adds a new node to the ring."""
        node = MerkleNode(data)
        self.nodes.append(node)

    def compute_root_hash(self) -> str:
        """Computes the Merkle root hash from the nodes in the ring."""
        hashes = [node.hash for node in self.nodes]
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:  # If odd number of hashes, duplicate the last one
                hashes.append(hashes[-1])
            hashes = [
                hashlib.sha256((hashes[i] + hashes[i + 1]).encode()).hexdigest()
                for i in range(0, len(hashes), 2)
            ]
        return hashes[0] if hashes else ""

    def export_to_toml(self) -> str:
        """Exports the Merkle ring to a TOML-formatted string."""
        toml_data = {
            "merkle_ring": [{"data": node.data, "hash": node.hash} for node in self.nodes],
            "root_hash": self.compute_root_hash()
        }
        return toml.dumps(toml_data)

    @staticmethod
    def import_from_toml(toml_string: str) -> 'MerkleRing':
        """Creates a MerkleRing instance from a TOML-formatted string."""
        data = toml.loads(toml_string)
        ring = MerkleRing()
        for node_info in data.get("merkle_ring", []):
            ring.add_node(node_info["data"])
        return ring

# Example usage:
if __name__ == "__main__":
    # Create a Merkle ring and add some nodes
    ring = MerkleRing()
    ring.add_node("Node 1 data")
    ring.add_node("Node 2 data")
    ring.add_node("Node 3 data")

    # Export to TOML
    toml_output = ring.export_to_toml()
    print("Exported TOML:")
    print(toml_output)

    # Import from TOML
    imported_ring = MerkleRing.import_from_toml(toml_output)
    print("\nImported Merkle Root Hash:", imported_ring.compute_root_hash())
