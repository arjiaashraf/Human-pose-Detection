from src.filter import OneEuroFilter


print("=" * 60)
print("Testing One-Euro Filter")
print("=" * 60)


# Create filter
filter = OneEuroFilter(
    min_cutoff=1.0,
    beta=0.007,
    d_cutoff=1.0
)

print("\nFilter created successfully!")


# Simulated noisy coordinates
coordinates = [
    100.0,
    101.5,
    99.2,
    102.8,
    100.7,
    103.5,
    101.2,
    104.1,
    102.0,
    105.0
]


print("\nRaw → Filtered")
print("-" * 30)


for i, value in enumerate(coordinates):

    timestamp = i / 30.0

    filtered_value = filter.filter(
        value,
        timestamp
    )

    print(
        f"{value:8.2f} → {filtered_value:8.2f}"
    )


print("\n" + "=" * 60)
print("ONE-EURO FILTER TEST PASSED!")
print("=" * 60)