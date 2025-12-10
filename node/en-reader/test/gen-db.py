import struct
import json

def make_kv_file(input_json, output_file):
    # input_json: dict {key: value}
    items = []
    def hash32(key):
        # Same 32-bit hash algorithm as in JavaScript
        h = 0
        for c in key:
            h = (h * 31 + ord(c)) & 0xffffffff
        return h
    
    for key, value in input_json.items():
        key_hash = hash32(key)
        items.append((key_hash, key, value))

    # sort by keyHash
    items.sort(key=lambda x: x[0])

    # write file
    with open(output_file, "wb") as f:
        N = len(items)
        f.write(struct.pack("<I", N))  # write count

        # prepare value area
        offsets = []
        data_blob = b""
        curr = 0

        for _, key, value in items:
            offsets.append(curr)
            value_bytes = value.encode("utf8")
            data_blob += value_bytes
            curr += len(value_bytes)

        # write index table
        for (key_hash, _, _), offset in zip(items, offsets):
            f.write(struct.pack("<II", key_hash, offset))

        # write values concatenated
        f.write(data_blob)

# -------- Example usage --------
data = {
    "apple": "fruit",
    "banana": "yellow",
    "cat": "animal"
}

make_kv_file(data, "kv.db")
print("kv.db created")