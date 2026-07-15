import numpy as np

from ai.analysis.pipeline import HealthPipeline

t = np.linspace(0,60,64*60)

signal = np.sin(2*np.pi*1.2*t)

signal += np.random.normal(0,0.2,len(signal))

pipeline = HealthPipeline()

result = pipeline.analyze_ppg(signal)

print(result)