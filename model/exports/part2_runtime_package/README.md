# Part 2 Runtime Package

This package was exported by Notebook 06.

Files:

- `best_checkpoint.pt`: trained R(2+1)D-18 catch/not-catch checkpoint.
- `runtime_config.json`: preprocessing, threshold, class mapping, and integration settings.
- `runtime_inference.py`: lightweight standalone action timeline inference module.

Basic usage:

```python
from runtime_inference import predict_action_timeline

timeline = predict_action_timeline(
    video_path="path/to/video.avi",
    package_dir="path/to/part2_runtime_package",
)
print(timeline.head())
```

Current checkpoint baseline:

- Best epoch: 6
- Validation F1: 0.64
- Validation recall: 0.5333333333333333
- Validation precision: 0.8

Important limitation:

The action model detects catch-like motion. Full process validation requires the application layer to combine this with object, hand, pose, and order/state rules.
