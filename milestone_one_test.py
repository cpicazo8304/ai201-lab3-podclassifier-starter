from classifier import load_labeled_examples

examples = load_labeled_examples()
print(f'{len(examples)} labeled examples loaded')
print([e['label'] for e in examples])