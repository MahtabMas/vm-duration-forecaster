from data import load_and_split

train_data, validation_data, test_data = load_and_split()

print("train:", len(train_data))
print("validation:", len(validation_data))
print("test:", len(test_data))