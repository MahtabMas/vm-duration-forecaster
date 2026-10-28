import pandas as pd
from data import load_and_split, make_target

train_data, validation_data, test_data = load_and_split()

# Days 0 to 11 only. The test rows stay sealed.
jobs = pd.concat([train_data, validation_data])
jobs["log_duration"] = make_target(jobs)

by_day = jobs.groupby("day").agg(
    number_of_jobs=("starttime", "size"),
    mean_log_duration=("log_duration", "mean"),
)
print(by_day.round(3))

print()
print("same weekday, one week apart (day d vs day d+7):")
for day in range(5):
    first = by_day.loc[day]
    second = by_day.loc[day + 7]
    print(f"  day {day:2d} vs day {day + 7:2d}: "
          f"jobs {int(first['number_of_jobs']):5d} vs {int(second['number_of_jobs']):5d}   "
          f"mean log duration {first['mean_log_duration']:.3f} vs {second['mean_log_duration']:.3f}")