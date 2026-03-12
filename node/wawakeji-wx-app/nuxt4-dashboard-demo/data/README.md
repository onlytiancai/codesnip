# ECDICT Dictionary Data

This directory contains the ECDICT dictionary data files.

## Download

Download the dictionary data from ECDICT GitHub repository:

```bash
# Download the 7z archive (49MB)
curl -L -o stardict.7z "https://github.com/skywind3000/ECDICT/raw/refs/heads/master/stardict.7z"
```

Source: https://github.com/skywind3000/ECDICT

## Extract

Extract the 7z archive (requires p7zip):

```bash
# Install p7zip if not installed
brew install p7zip

# Extract the archive
7z x stardict.7z -y
```

This will extract `stardict.csv` (~222MB) containing 3.4M+ word entries.

## Import to SQLite Database

Import the CSV directly into the dictionary database:

```bash
# Reset the dictionary database (optional)
rm -f ../prisma/dictionary.db
pnpm prisma db push --schema=../prisma/dictionary.prisma --skip-generate

# Import CSV to temp table
cd ..
sqlite3 prisma/dictionary.db << 'EOF'
.mode csv
.import data/stardict.csv temp_import
EOF

# Insert into Dictionary table
sqlite3 prisma/dictionary.db << 'EOF'
INSERT OR IGNORE INTO Dictionary (word, phonetic, definition, translation, pos, collins, oxford, tag, bnc, frq, exchange)
SELECT
  word,
  NULLIF(phonetic, '') as phonetic,
  NULLIF(definition, '') as definition,
  NULLIF(translation, '') as translation,
  NULLIF(pos, '') as pos,
  CASE WHEN collins = '' THEN NULL ELSE CAST(collins AS INTEGER) END as collins,
  CASE WHEN oxford = '' THEN NULL ELSE CAST(oxford AS INTEGER) END as oxford,
  NULLIF(tag, '') as tag,
  CASE WHEN bnc = '' THEN NULL ELSE CAST(bnc AS INTEGER) END as bnc,
  CASE WHEN frq = '' THEN NULL ELSE CAST(frqt AS INTEGER) END as frq,
  NULLIF(exchange, '') as exchange
FROM temp_import
WHERE word != 'word';

DROP TABLE temp_import;
VACUUM;
EOF
```

## CSV Format

| Column | Description |
|--------|-------------|
| word | The word (lowercase) |
| phonetic | IPA phonetic transcription |
| definition | English definition |
| translation | Chinese translation |
| pos | Part of speech |
| collins | Collins star rating (1-5) |
| oxford | Oxford 3000 flag (0/1) |
| tag | Tags (zk=middle school, gk=high school, cet4/6, ielts, etc.) |
| bnc | British National Corpus frequency |
| frq | Contemporary Corpus frequency |
| exchange | Word forms (p:past/i:ing/d:past_part/3:3rd_person/s:plural) |
| detail | Additional details |
| audio | Audio reference |

## Stats

- Total words: 3,402,563
- Words with phonetic: 353,133
- Words with translation: 3,386,027
- Database size: ~396MB

## Cleanup

After import, you can delete the downloaded files to save space:

```bash
rm -f stardict.7z stardict.csv
```

The dictionary database is located at `../prisma/dictionary.db`.