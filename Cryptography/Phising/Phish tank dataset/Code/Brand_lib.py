
import os
import re
import pandas as pd
import tldextract
from urllib.parse import urlparse

# ==========================================
# 1. Directory configuration
# ==========================================

base_dir = r"C:\Users\Lenovo\Desktop\Cryptography\Phising\Phish tank dataset"

input_file = os.path.join(
    base_dir,
    "tranco_746WX.csv"
)

output_dir = os.path.join(
    base_dir,
    "Legitimate Domain Library"
)

os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(
    output_dir,
    "brand_lib.csv"
)

# ==========================================
# 2. CSV configuration
# ==========================================

URL_COLUMN = "url"

CHUNK_SIZE = 50000

# Use the bundled PSL snapshot.
# No network download is required during extraction.
extractor = tldextract.TLDExtract(
    suffix_list_urls=None
)

# ==========================================
# 3. Domain extraction function
# ==========================================

def extract_domain(url):

    try:

        if pd.isna(url):
            return None

        url = str(url).strip().lower()

        if not url:
            return None

        # Add scheme if missing
        if not re.match(r"^[a-z][a-z0-9+.-]*://", url):
            url = "http://" + url

        parsed = urlparse(url)

        hostname = parsed.hostname

        if not hostname:
            return None

        # Remove trailing dot
        hostname = hostname.rstrip(".")

        # Extract registered domain
        result = extractor(hostname)

        if not result.domain or not result.suffix:
            return None

        registered_domain = (
            result.domain + "." + result.suffix
        )

        return registered_domain

    except Exception:
        return None


# ==========================================
# 4. Process CSV in chunks
# ==========================================

print("Starting domain extraction...")

unique_domains = set()

total_urls = 0
valid_domains = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        input_file,
        usecols=[URL_COLUMN],
        chunksize=CHUNK_SIZE,
        dtype=str
    ),
    start=1
):

    total_urls += len(chunk)

    # Extract domains
    domains = chunk[URL_COLUMN].apply(
        extract_domain
    )

    valid = domains.dropna()

    valid_domains += len(valid)

    unique_domains.update(valid)

    print(
        f"Chunk {chunk_number} processed | "
        f"URLs: {total_urls:,} | "
        f"Unique domains: {len(unique_domains):,}"
    )


# ==========================================
# 5. Create unique domain DataFrame
# ==========================================

domain_df = pd.DataFrame({

    "registered_domain": sorted(unique_domains)

})

# Extract domain name separately
domain_df["domain_name"] = (
    domain_df["registered_domain"]
    .str.split(".")
    .str[0]
)

# ==========================================
# 6. Save domain library
# ==========================================

domain_df.to_csv(
    output_file,
    index=False
)

# ==========================================
# 7. Save extraction statistics
# ==========================================

statistics = {

    "Total_URLs_Processed": total_urls,

    "Valid_Domain_Extractions": valid_domains,

    "Unique_Registered_Domains": len(unique_domains),

    "Duplicate_Extractions": (
        valid_domains - len(unique_domains)
    ),

    "Output_File": output_file

}

stats_file = os.path.join(
    output_dir,
    "extraction_statistics.txt"
)

with open(stats_file, "w") as f:

    for key, value in statistics.items():

        f.write(f"{key}: {value}\n")


# ==========================================
# 8. Display final results
# ==========================================

print("\n======================================")
print("DOMAIN EXTRACTION COMPLETED")
print("======================================")

print("Total URLs processed:", f"{total_urls:,}")

print("Valid domain extractions:", f"{valid_domains:,}")

print("Unique domains:", f"{len(unique_domains):,}")

print("\nFirst 20 domains:")

print(domain_df.head(20).to_string(index=False))

print("\nSaved files:")

print(output_file)

print(stats_file)

print("\n======================================")