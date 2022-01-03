import time

from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k

# Load the MovieLens 100k dataset. Only five
# star ratings are treated as positive.
data = fetch_movielens(min_rating=3.0)

# Instantiate and train the model
model = LightFM(loss="warp")

print("Training model...")
start_time = time.time()

# I evaluated the speedup on 3000 epochs.
# With 16 threads, the training takes about 66 seconds.
# With only 1 thread, it takes about 95 seconds.
# Speedup might be small because of size of dataset.
model.fit(data["train"], epochs=500, num_threads=16)
model.fit_partial

end_time = time.time()
print(f"Training finished! Elapsed time: {end_time - start_time:.2f} s.")

# Evaluate the trained model
test_precision = precision_at_k(model, data["test"], k=5).mean()

print(f"Result test precision: {test_precision}")
