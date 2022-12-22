from collections import defaultdict

def duplicate_oversample(data):
    # Create dict counter for each label
    label_sexist_counter = defaultdict(int)

    for sample in data:
        label_sexist_counter[sample['label_sexist']] += 1

    # Get the max count
    max_count = max(label_sexist_counter.values())

    # Create dict of lists for each label
    label_sexist_data = defaultdict(list)

    for sample in data:
        label_sexist_data[sample['label_sexist']].append(sample)

    # Duplicate the minority class samples to match the majority class
    for label, count in label_sexist_counter.items():
        if count < max_count:
            label_sexist_data[label] *= (max_count // count)
    
    # Concatenate the lists
    oversampled_data = []
    for label, data in label_sexist_data.items():
        oversampled_data += data

    return oversampled_data