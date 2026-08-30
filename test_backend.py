from backend import experiment_manager

print("--- CREATE ---")
new_exp = experiment_manager.create_experiment(
    name="Test Run",
    description="Verifying CRUD",
    steps=[
        {"instruction": "Do step one", "expected_action": "ACTION_1"},
        {"instruction": "Do step two", "expected_action": "ACTION_2"},
    ],
)
print(new_exp)

print("\n--- Reload check ---")
reloaded = experiment_manager.get_experiment(new_exp["id"])
assert reloaded["name"] == "Test Run"
print("Persistence OK ->", reloaded["name"])

print("\n--- DELETE ---")
experiment_manager.delete_experiment(new_exp["id"])
print("Cleaned up test experiment")

print("\nBLOCK 2 + 3 CHECKS PASSED")