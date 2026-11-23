
import os
import tldextract

extractor = tldextract.TLDExtract(suffix_list_urls=None)

# Obtain the same bundled suffix rules used by the training script
suffix_data = extractor._get_tld_extractor()
rules = suffix_data.tlds()

output_dir = (
    r"C:\Users\Lenovo\AndroidStudioProjects"
    r"\DeviceThreatDetection\app\src\main\assets"
)

os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "public_suffix_list.dat")

with open(output_file, "w", encoding="utf-8") as f:
    for rule in sorted(rules):
        f.write(rule + "\n")

print("Public suffix list exported.")
print("Number of rules:", len(rules))
print("Saved to:", output_file)
