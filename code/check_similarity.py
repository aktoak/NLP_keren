import sys
import re

def strip_tags(input_path, output_path):
    """
    Remove _TAG suffixes from tokens and write clean text to output_path.
    Example: "Dog_NOUN eats_VERB" -> "Dog eats"
    """
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            tokens = line.strip().split()
            # Remove last "_TAG" part from each token
            words = [re.sub(r'_(?=[^_]+$)', ' ', token).rsplit(' ', 1)[0] for token in tokens]
            cleaned = [w.replace('_', '') for w in words]
            outfile.write(" ".join(cleaned) + "\n")
    print(f"✅ Cleaned text written to: {output_path}")

def compare_wtag_files(file1, file2):
    """
    Compare two .wtag files line by line, token by token.
    Report mismatched lines or tokens.
    """
    print(f"📎 Comparing files:\n  - {file1}\n  - {file2}")
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    if len(lines1) != len(lines2):
        print(f"❌ File lengths differ: {len(lines1)} vs {len(lines2)} lines")
        return

    mismatch_count = 0
    for i, (line1, line2) in enumerate(zip(lines1, lines2), start=1):
        tokens1 = line1.strip().split()
        tokens2 = line2.strip().split()

        if len(tokens1) != len(tokens2):
            print(f"❌ Line {i} token count mismatch: {len(tokens1)} vs {len(tokens2)}")
            mismatch_count += 1
            continue

        for j, (tok1, tok2) in enumerate(zip(tokens1, tokens2)):
            if tok1 != tok2:
                print(f"🔁 Mismatch on line {i}, token {j + 1}: '{tok1}' != '{tok2}'")
                mismatch_count += 1

    if mismatch_count == 0:
        print("✅ Files match exactly.")
    else:
        print(f"⚠️ Found {mismatch_count} mismatched tokens.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Utility to strip _TAGs or compare .wtag files.")
    parser.add_argument("--strip", nargs=2, metavar=('input.wtag', 'output.txt'), help="Strip tags from a .wtag file")
    parser.add_argument("--compare", nargs=2, metavar=('file1.wtag', 'file2.wtag'), help="Compare two .wtag files")

    args = parser.parse_args()

    if args.strip:
        strip_tags(args.strip[0], args.strip[1])

    if args.compare:
        compare_wtag_files(args.compare[0], args.compare[1])

    if not args.strip and not args.compare:
        print("⚠️ No action specified. Use --strip or --compare.")
