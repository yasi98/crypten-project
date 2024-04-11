import typing

import crypten

def crypten_collate(batch):
    # Extract the first element of the batch to determine its type
    elem = batch[0]
    elem_type = type(elem)

    # If the element is a CrypTensor, stack all tensors along a new dimension
    if isinstance(elem, crypten.CrypTensor):
        return crypten.stack(list(batch), dim=0)

    # If the element is a sequence (e.g., list, tuple)
    elif isinstance(elem, typing.Sequence):
        # Ensure that all elements in the batch have equal size
        size = len(elem)
        assert all(len(b) == size for b in batch), "each element in list of batch should be of equal size"
        # Transpose the batch and apply crypten_collate recursively to each element
        transposed = zip(*batch)
        return [crypten_collate(samples) for samples in transposed]

    # If the element is a mapping (e.g., dictionary)
    elif isinstance(elem, typing.Mapping):
        # Apply crypten_collate recursively to each value in the dictionary
        return {key: crypten_collate([b[key] for b in batch]) for key in elem}

    # If the element is a namedtuple
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        # Recursively call crypten_collate on each element of the transposed batch
        return elem_type(*(crypten_collate(samples) for samples in zip(*batch)))

    # If the element type is not recognized, return an error message
    return "crypten_collate: batch must contain CrypTensor, dicts or lists; found {}".format(elem_type)
