from jax import devices, device_put, device_put_replicated
import jax.numpy as jnp
import tensorflow_datasets as tfds

gpus = devices('gpu')
cpus = devices('cpu')

def put_on_device(data_set, devices, id=None):
  for key in data_set:
    if key in ['image', 'label']:
      data_set[key] = device_put_replicated(data_set[key], devices) if id is None else device_put(data_set[key], devices[id])

  return data_set  


def load_data(name, batch_size=-1, platform='cpu', id=None):
  """Load train and test datasets into memory."""
  ds_builder = tfds.builder(name)
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=batch_size))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=batch_size))

  train_ds = put_on_device(train_ds, devices(platform), id=id)
  test_ds = put_on_device(test_ds, devices(platform), id=id)
  
  return train_ds, test_ds