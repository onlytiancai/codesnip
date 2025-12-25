import struct
import json
import sqlite3

def make_kv_file_from_sqlite(db_path, output_file):
    def hash32(key):
        # Same 32-bit hash algorithm as in JavaScript
        h = 0
        for c in key:
            h = (h * 31 + ord(c)) & 0xffffffff
        return h
    
    # First pass: Read all words, calculate hashes, and collect them
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    cursor = conn.cursor()
    
    print("First pass: Collecting words and calculating hashes...")
    
    # Get total count first
    cursor.execute("SELECT COUNT(*) FROM stardict")
    total_count = cursor.fetchone()[0]
    print(f"Total words to process: {total_count}")
    
    # Now fetch all words with progress tracking
    cursor.execute("SELECT word FROM stardict")
    
    # Collect (hash, word) pairs
    word_hashes = []
    processed = 0
    
    for (word,) in cursor.fetchall():
        key_hash = hash32(word)
        word_hashes.append((key_hash, word))
        
        processed += 1
        if processed % 1000 == 0 or processed == total_count:
            progress = (processed / total_count) * 100
            print(f"Progress: {processed}/{total_count} ({progress:.1f}%)")
    
    # Sort by key hash
    word_hashes.sort(key=lambda x: x[0])
    
    print(f"Total items: {len(word_hashes)}")
    
    # Second pass: Write the file
    print("Second pass: Writing data to file...")
    
    with open(output_file, "wb") as f:
        N = len(word_hashes)
        f.write(struct.pack("<I", N))  # Write count
        
        # Step 1: Write placeholder for index table (will be updated later)
        index_table_offset = f.tell()
        # Write empty index table (8 bytes per entry)
        f.write(b'\x00' * 8 * N)
        
        # Step 2: Write value area and record offsets
        offsets = []
        curr_offset = 0
        
        processed = 0
        
        for _, word in word_hashes:
            # Read the full row for this word
            cursor.execute("SELECT * FROM stardict WHERE word = ?", (word,))
            row = cursor.fetchone()
            
            if row:
                # Convert row to dict and then to JSON string
                row_dict = dict(row)
                value = json.dumps(row_dict, ensure_ascii=False)
                value_bytes = value.encode("utf8")
                
                # Record offset
                offsets.append(curr_offset)
                
                # Write value
                f.write(value_bytes)
                curr_offset += len(value_bytes)
            
            processed += 1
            if processed % 1000 == 0 or processed == N:
                progress = (processed / N) * 100
                print(f"Writing values: {processed}/{N} ({progress:.1f}%)")
        
        # Step 3: Update index table
        print("Updating index table...")
        f.seek(index_table_offset)
        
        total = len(word_hashes)
        processed = 0
        
        for (key_hash, _), offset in zip(word_hashes, offsets):
            f.write(struct.pack("<II", key_hash, offset))
            
            processed += 1
            if processed % 1000 == 0 or processed == total:
                progress = (processed / total) * 100
                print(f"Updating index: {processed}/{total} ({progress:.1f}%)")
    
    conn.close()
    print(f"{output_file} created from {db_path}")

# -------- Example usage --------
db_path = "/Users/huhao/Downloads/stardict.db"
output_file = "dict.db"

make_kv_file_from_sqlite(db_path, output_file)