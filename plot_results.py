import matplotlib.pyplot as plt
import re
def parse_results(filename="Results.txt"):
    results = []
    with open(filename, "r") as f:
        content = f.read().strip().split("\n\n")
    for block in content:
        lines = block.strip().split("\n")
        if len(lines) < 4:  
            continue
        model_name = lines[0].strip()
        precision = float(re.search(r"[-+]?\d*\.\d+|\d+", lines[1]).group())
        recall = float(re.search(r"[-+]?\d*\.\d+|\d+", lines[2]).group())
        f1 = float(re.search(r"[-+]?\d*\.\d+|\d+", lines[3]).group())
        results.append((model_name, precision, recall, f1))
    return results
def plot_results(results):
    labels = ["Precision", "Recall", "F1 Score"]
    plt.figure(figsize=(8,6))
    for i, (model, p, r, f1) in enumerate(results):
        scores = [p, r, f1]
        plt.bar([x + i*0.25 for x in range(len(scores))],
                scores, width=0.25, label=model)
    plt.xticks([x + 0.25 for x in range(len(labels))], labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Landslide Detection Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()
if __name__ == "__main__":
    results = parse_results("Results.txt")
    if results:
        print("Parsed Results:", results)
        plot_results(results)
    else:
        print(" No valid results found in Results.txt")
